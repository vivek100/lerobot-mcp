"""Talk mode: the gripper as a mouth synced to speech, head nods at sentence ends, mood gestures.

The mouth follows either the loudness envelope of synthesized speech (Windows SAPI voices, played on
the control clock when the motion starts) or, without voice, a syllable-rate estimate from the text.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import wave
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from so101_mcp.runtime.animation import GRIP, LIFT, PAN, PRESENT, WFLEX, WROLL, Offsets, combine, hold, oscillate

Mood = Literal["neutral", "happy", "excited", "sad", "curious"]

SYLLABLE_S = 0.19
WORD_GAP_S = 0.05
PAUSE_S = {",": 0.25, ";": 0.3, ":": 0.3, ".": 0.45, "!": 0.45, "?": 0.45}
SENTENCE_END = ".!?"
MOUTH_OPEN = 45.0  # gripper units at full loudness
MOUTH_LEAD_S = 0.08  # servos lag the command; open the mouth slightly ahead of the sound
MAX_SPEECH_S = 100.0

MOOD_BODY: dict[str, tuple[dict[int, float], dict[int, tuple[float, ...]]]] = {
    "neutral": ({}, {PAN: (4, 0.2), LIFT: (1.5, 0.3)}),
    "happy": ({}, {PAN: (7, 0.45), WROLL: (8, 0.9), LIFT: (2.5, 0.9)}),
    "excited": ({}, {PAN: (10, 0.7), WROLL: (10, 1.4), LIFT: (3, 1.4), WFLEX: (4, 1.4)}),
    "sad": ({WFLEX: 20, LIFT: 6}, {PAN: (3, 0.15)}),
    "curious": ({WROLL: 15, WFLEX: -6}, {PAN: (4, 0.25)}),
}


def talk_base() -> np.ndarray:
    base = PRESENT.copy()
    base[GRIP] = 5.0  # mouth closed (not squeezed shut)
    return base


def count_syllables(word: str) -> int:
    letters = re.sub(r"[^a-z]", "", word.lower())
    if not letters:
        return len(re.sub(r"\D", "", word))  # digits: roughly one beat each
    count = len(re.findall(r"[aeiouy]+", letters))
    if letters.endswith("e") and count > 1 and not letters.endswith(("le", "ee")):
        count -= 1
    return max(count, 1)


def text_mouth(text: str, fps: int) -> tuple[np.ndarray, list[float]]:
    """Mouth opening 0-1 per tick from syllable timing, plus sentence-end times."""
    rng = np.random.default_rng(sum(map(ord, text)))
    pulses: list[tuple[float, float, float]] = []
    sentence_ends: list[float] = []
    t = 0.0
    for token in re.findall(r"[\w']+|[.,!?;:]", text):
        if token in PAUSE_S:
            if token in SENTENCE_END:
                sentence_ends.append(t)
            t += PAUSE_S[token]
            continue
        for _ in range(count_syllables(token)):
            pulses.append((t, SYLLABLE_S, float(rng.uniform(0.55, 1.0))))
            t += SYLLABLE_S
        t += WORD_GAP_S
    if not sentence_ends or sentence_ends[-1] < t - 0.5:
        sentence_ends.append(t)
    times = np.arange(int(np.ceil(t * fps)) + 1) / fps
    mouth = np.zeros(len(times))
    for start, duration, amp in pulses:
        s = (times - start) / duration
        inside = (s >= 0) & (s <= 1)
        mouth[inside] = np.maximum(mouth[inside], amp * np.sin(np.pi * s[inside]))
    return mouth, sentence_ends


def audio_mouth(path: str, fps: int) -> tuple[np.ndarray, list[float]]:
    """Mouth opening 0-1 per tick from the speech loudness; pauses longer than 0.3 s count as sentence ends."""
    with wave.open(path) as w:
        rate = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    hop = rate // fps
    frames = len(audio) // hop
    rms = np.sqrt(np.mean(audio[: frames * hop].reshape(frames, hop) ** 2, axis=1))
    loud = rms / max(np.percentile(rms, 95), 1.0)
    mouth = np.clip((loud - 0.08) / 0.92, 0.0, 1.0)
    quiet = mouth < 0.05
    sentence_ends, run = [], 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run == int(0.3 * fps):
            sentence_ends.append((i - run + 1) / fps)
    return np.append(mouth, 0.0), sentence_ends


def slew(series: np.ndarray, max_step: float) -> np.ndarray:
    """Rate-limit a signal (forward then backward pass, so peaks are lowered symmetrically, not delayed)."""
    out = series.copy()
    for i in range(1, len(out)):
        out[i] = np.clip(out[i], out[i - 1] - max_step, out[i - 1] + max_step)
    for i in range(len(out) - 2, -1, -1):
        out[i] = np.clip(out[i], out[i + 1] - max_step, out[i + 1] + max_step)
    return out


def nods(sentence_ends: list[float], duration: float) -> Offsets:
    """A small head dip after each sentence."""

    def offsets(t: np.ndarray) -> np.ndarray:
        out = np.zeros((len(t), 6))
        for end in sentence_ends:
            s = (t - end) / 0.6
            inside = (s >= 0) & (s <= 1)
            out[inside, WFLEX] += 8.0 * np.sin(np.pi * s[inside]) ** 2
        return out

    return offsets


@dataclass
class TalkScript:
    """Everything the control thread needs to play one utterance."""

    text: str
    mood: str
    fps: int
    mouth: np.ndarray  # 0-1 per tick
    sentence_ends: list[float]
    audio_path: str | None = None
    voice_name: str | None = None
    mute: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return (len(self.mouth) - 1) / self.fps

    def offsets(self, gripper_units_per_s: float) -> Offsets:
        lead = int(round(MOUTH_LEAD_S * self.fps))
        mouth = np.append(self.mouth[lead:], np.zeros(lead)) * MOUTH_OPEN
        mouth = slew(mouth, 0.85 * gripper_units_per_s / self.fps)
        times = np.arange(len(mouth)) / self.fps
        tilt, sway = MOOD_BODY[self.mood]
        body = combine(hold(self.duration, tilt, ramp=0.6), oscillate(self.duration, sway, ramp=0.6),
                       nods(self.sentence_ends, self.duration))

        def offsets(t: np.ndarray) -> np.ndarray:
            out = body(t)
            out[:, GRIP] += np.interp(t, times, mouth)
            return out

        return offsets

    def play(self) -> None:
        if self.audio_path and not self.mute:
            import winsound

            winsound.PlaySound(self.audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def finish(self) -> None:
        if not self.audio_path:
            return
        if not self.mute:
            import winsound

            winsound.PlaySound(None, 0)
        try:
            os.remove(self.audio_path)
        except OSError:
            pass


_tts_lock = threading.Lock()


def synthesize(text: str, voice: str | None = None) -> tuple[str, str]:
    """Speak ``text`` into a temporary 22 kHz mono WAV with a Windows SAPI voice; returns (path, voice name)."""
    import pythoncom
    import win32com.client

    with _tts_lock:
        pythoncom.CoInitialize()
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            voices = speaker.GetVoices()
            names = [voices.Item(i).GetDescription() for i in range(voices.Count)]
            if voice:
                matches = [i for i, n in enumerate(names) if voice.lower() in n.lower()]
                if not matches:
                    raise ValueError(f"No voice like {voice!r}; installed: {names}")
                speaker.Voice = voices.Item(matches[0])
            chosen = speaker.Voice.GetDescription()
            fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
            fmt.Type = 22  # SAFT22kHz16BitMono
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Format = fmt
            fd, path = tempfile.mkstemp(prefix="so101_talk_", suffix=".wav")
            os.close(fd)
            stream.Open(path, 3, False)  # SSFMCreateForWrite
            speaker.AudioOutputStream = stream
            speaker.Speak(text)
            stream.Close()
            return path, chosen
        finally:
            pythoncom.CoUninitialize()


def prepare_talk(text: str, mood: str, fps: int, voice: bool, voice_name: str | None, mute: bool) -> TalkScript:
    """Build the mouth track (and audio) on the caller's thread; the control thread only plays it."""
    text = text.strip()
    if not text:
        raise ValueError("text is empty")
    if mood not in MOOD_BODY:
        raise ValueError(f"Unknown mood {mood!r}; choose from {list(MOOD_BODY)}")
    if voice:
        path, chosen = synthesize(text, voice_name)
        mouth, ends = audio_mouth(path, fps)
        script = TalkScript(text, mood, fps, mouth, ends, audio_path=path, voice_name=chosen, mute=mute)
    else:
        mouth, ends = text_mouth(text, fps)
        script = TalkScript(text, mood, fps, mouth, ends)
    if script.duration > MAX_SPEECH_S:
        raise ValueError(f"Speech would last {script.duration:.0f}s; split it into parts under {MAX_SPEECH_S:.0f}s")
    return script
