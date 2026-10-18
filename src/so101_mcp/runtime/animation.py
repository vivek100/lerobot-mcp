"""Expressive motion: the animation library, fitting to the arm's speed budget, and playback.

An animation is a base pose plus a function of time giving offsets from it, zero at both ends. It is
sampled at the control rate and fitted to the backend's speed and acceleration budget: each joint's
amplitude shrinks first (the tempo is what makes a gesture readable), and only if a joint would shrink
below ``min_amplitude`` is the whole animation slowed down. Speech-synced animations never slow down.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from so101_mcp.joints import JOINTS
from so101_mcp.runtime.behaviors import Behavior, BehaviorOutcome, min_jerk_duration

PAN, LIFT, ELBOW, WFLEX, WROLL, GRIP = range(6)

# Arm raised, gripper pointing at the viewer like a desk lamp's head; the jaws open up/down like a mouth.
PRESENT = np.array([0.0, -45.0, 35.0, 15.0, 0.0, 20.0])
# Forearm raised and gripper open, like a hand held up to wave.
WAVE_POSE = np.array([0.0, -10.0, -40.0, -30.0, 0.0, 60.0])

Offsets = Callable[[np.ndarray], np.ndarray]  # times (N,) -> offsets from the base pose (N, 6)


def smooth(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def envelope(t: np.ndarray, duration: float, ramp: float) -> np.ndarray:
    """0 -> 1 -> 0 with smooth ramps, so every animation starts and ends at rest on its base pose."""
    return smooth(np.minimum(t / ramp, (duration - t) / ramp))


def oscillate(duration: float, joints: dict[int, tuple[float, ...]], ramp: float = 0.4) -> Offsets:
    """Sum of sines: joints map index -> (amplitude, hz) or (amplitude, hz, phase_deg)."""

    def offsets(t: np.ndarray) -> np.ndarray:
        out = np.zeros((len(t), len(JOINTS)))
        for j, (amp, hz, *phase) in joints.items():
            out[:, j] = amp * np.sin(2 * np.pi * hz * t + np.radians(phase[0] if phase else 0.0))
        return out * envelope(t, duration, ramp)[:, None]

    return offsets


def hold(duration: float, joints: dict[int, float], ramp: float = 0.6) -> Offsets:
    """Constant offsets eased in and out (a tilt or droop held for the whole animation)."""
    vector = np.zeros(len(JOINTS))
    for j, value in joints.items():
        vector[j] = value
    return lambda t: envelope(t, duration, ramp)[:, None] * vector


def keyframes(points: list[tuple[float, dict[int, float]]]) -> Offsets:
    """Smooth stop-and-go between keyed offsets; a key carries forward joints it does not mention."""
    times = np.array([p[0] for p in points], dtype=np.float64)
    values = np.zeros((len(points), len(JOINTS)))
    current = np.zeros(len(JOINTS))
    for i, (_, keyed) in enumerate(points):
        current = current.copy()
        for j, value in keyed.items():
            current[j] = value
        values[i] = current

    def offsets(t: np.ndarray) -> np.ndarray:
        t = np.clip(t, times[0], times[-1])
        i = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
        s = smooth((t - times[i]) / (times[i + 1] - times[i]))
        return values[i] + s[:, None] * (values[i + 1] - values[i])

    return offsets


def combine(*parts: Offsets) -> Offsets:
    return lambda t: sum(part(t) for part in parts)


@dataclass(frozen=True)
class Animation:
    name: str
    description: str
    base: np.ndarray
    cycle_s: float
    make: Callable[[float], Offsets]  # total seconds -> offsets
    loopable: bool = False  # an ongoing state (express) rather than a one-shot gesture


def _repeating(cycle: float, points: list[tuple[float, dict[int, float]]]) -> Callable[[float], Offsets]:
    keyed = keyframes(points)
    return lambda total: (lambda t: keyed(np.where(t >= total, cycle, np.mod(t, cycle))))


def _library() -> dict[str, Animation]:
    animations = [
        Animation("nod", "Nod yes: the head (gripper) dips and rises.", PRESENT, 2.4,
                  lambda d: oscillate(d, {WFLEX: (12, 1.25), LIFT: (2, 1.25)}, ramp=0.3)),
        Animation("shake", "Shake no: the base turns side to side.", PRESENT, 2.8,
                  lambda d: oscillate(d, {PAN: (10, 0.9), WROLL: (4, 0.9)}, ramp=0.3)),
        Animation("wave", "Wave hello with the forearm raised and the gripper open.", WAVE_POSE, 3.2,
                  lambda d: oscillate(d, {PAN: (10, 0.8), WFLEX: (15, 0.8, 90)})),
        Animation("tada", "Presentation 'ta-da': crouch, then spring up with the gripper wide open.", PRESENT, 3.0,
                  _repeating(3.0, [
                      (0.0, {}),
                      (0.5, {LIFT: 8, ELBOW: 8, WFLEX: 15, GRIP: -20}),
                      (1.1, {LIFT: -12, ELBOW: -20, WFLEX: -25, GRIP: 75}),
                      (2.3, {}), (3.0, {LIFT: 0, ELBOW: 0, WFLEX: 0, GRIP: 0}),
                  ])),
        Animation("bow", "Bow forward, hold, and come back up.", PRESENT, 2.8,
                  _repeating(2.8, [
                      (0.0, {}),
                      (1.0, {LIFT: 15, ELBOW: 5, WFLEX: 25, GRIP: -15}),
                      (1.8, {}), (2.8, {LIFT: 0, ELBOW: 0, WFLEX: 0, GRIP: 0}),
                  ])),
        Animation("shrug", "Shrug: lift, tilt the head and open the gripper, then settle.", PRESENT, 1.8,
                  _repeating(1.8, [
                      (0.0, {}),
                      (0.45, {LIFT: -6, WFLEX: -8, WROLL: 15, GRIP: 35}),
                      (1.2, {}), (1.8, {LIFT: 0, WFLEX: 0, WROLL: 0, GRIP: 0}),
                  ])),
        Animation("point", "Point at the viewer: reach forward with the gripper closed, then return.", PRESENT, 2.6,
                  _repeating(2.6, [
                      (0.0, {}),
                      (0.9, {LIFT: 15, ELBOW: -25, WFLEX: -10, GRIP: -20}),
                      (1.7, {}), (2.6, {LIFT: 0, ELBOW: 0, WFLEX: 0, GRIP: 0}),
                  ])),
        Animation("look_around", "Look left, then right, then back to the viewer.", PRESENT, 5.4,
                  _repeating(5.4, [
                      (0.0, {}),
                      (1.2, {PAN: -35, WROLL: -10}), (2.0, {}),
                      (3.4, {PAN: 35, WROLL: 10}), (4.2, {}),
                      (5.4, {PAN: 0, WROLL: 0}),
                  ])),
        Animation("dance", "A short dance: sway, roll and clap the gripper to the beat.", PRESENT, 6.0,
                  lambda d: oscillate(d, {
                      PAN: (18, 0.5), WROLL: (20, 1.0), LIFT: (5, 1.0, 90), WFLEX: (10, 1.0), GRIP: (20, 2.0),
                  }, ramp=0.6)),
        Animation("excited", "Excited: quick little nods with the gripper chattering.", PRESENT, 2.5,
                  lambda d: oscillate(d, {WFLEX: (8, 2.2), GRIP: (18, 2.2), LIFT: (3, 2.2)}, ramp=0.3)),
        # Ongoing states for express(): eased in, held, eased out.
        Animation("idle", "Idle: slow breathing motion.", PRESENT, 5.0,
                  lambda d: oscillate(d, {LIFT: (2, 0.22), WFLEX: (3, 0.22, 60), PAN: (4, 0.08)}, ramp=1.0),
                  loopable=True),
        Animation("listening", "Listening: head tilted toward the speaker, small attentive nods.", PRESENT, 4.0,
                  lambda d: combine(hold(d, {WROLL: 12, WFLEX: -5}), oscillate(d, {WFLEX: (3, 0.4)}, ramp=1.0)),
                  loopable=True),
        Animation("thinking", "Thinking: look up and away with the head tilted, drifting slowly.", PRESENT, 4.0,
                  lambda d: combine(hold(d, {WROLL: 22, WFLEX: -12, PAN: -12}),
                                    oscillate(d, {PAN: (5, 0.1), GRIP: (5, 0.3)}, ramp=1.0)),
                  loopable=True),
        Animation("happy", "Happy: a gentle bouncy sway.", PRESENT, 4.0,
                  lambda d: oscillate(d, {PAN: (8, 0.5), WROLL: (10, 1.0), LIFT: (3, 1.0)}, ramp=0.6),
                  loopable=True),
        Animation("sad", "Sad: head drooped, slow sway.", PRESENT, 4.0,
                  lambda d: combine(hold(d, {WFLEX: 30, LIFT: 8, GRIP: -15}, ramp=1.0),
                                    oscillate(d, {PAN: (3, 0.1)}, ramp=1.0)),
                  loopable=True),
    ]
    return {a.name: a for a in animations}


LIBRARY: dict[str, Animation] = _library()


@dataclass
class FittedMotion:
    samples: np.ndarray  # (N, 6) absolute poses at the control rate
    fps: int
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return (len(self.samples) - 1) / self.fps


def fit_motion(
    base: np.ndarray,
    offsets: Offsets,
    duration: float,
    fps: int,
    lower: np.ndarray,
    upper: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray | None = None,
    min_amplitude: float = 0.5,
    allow_slowdown: bool = True,
    margin: float = 0.9,
) -> FittedMotion:
    """Sample an animation and scale it to the per-joint speed (and acceleration) budget."""

    def sample(stretch: float) -> np.ndarray:
        n = int(round(duration * stretch * fps)) + 1
        return offsets(np.minimum(np.arange(n) / (fps * stretch), duration))

    raw = sample(1.0)
    need_v = np.abs(np.diff(raw, axis=0)).max(axis=0) * fps / (margin * velocity)
    need_a = np.zeros_like(need_v)
    if acceleration is not None and len(raw) > 2:
        need_a = np.abs(np.diff(raw, 2, axis=0)).max(axis=0) * fps**2 / (margin * acceleration)
    need = np.maximum(need_v, need_a)
    scale = np.where(need > 1.0, 1.0 / np.maximum(need, 1e-9), 1.0)
    if allow_slowdown:
        scale = np.maximum(scale, min_amplitude)
    stretch = 1.0
    if allow_slowdown:
        stretch = max(1.0, float((need_v * scale).max()), float(np.sqrt((need_a * scale).max())))

    notes = []
    for j in np.flatnonzero(scale < 0.99):
        notes.append(f"{JOINTS[j]} amplitude scaled to {scale[j]:.2f} to fit the speed limits")
    if stretch > 1.01:
        notes.append(f"slowed {stretch:.2f}x to fit the speed limits")
    samples = base + sample(stretch) * scale
    clipped = np.clip(samples, lower, upper)
    if np.any(np.abs(clipped - samples) > 1e-6):
        notes.append("clamped_to_joint_limits")
    return FittedMotion(clipped, fps, notes)


class TrajectoryBehavior(Behavior):
    """Glide from the commanded pose to the motion's first sample, play it on the control clock, settle.

    ``on_play`` fires on the tick playback starts (used to start speech audio in sync);
    ``on_finish`` fires when the job ends for any reason (used to stop that audio).
    """

    def __init__(
        self,
        kind: str,
        motion: FittedMotion,
        max_speed: np.ndarray,
        tolerance: float = 2.0,
        settle_timeout: float = 1.5,
        on_play: Callable[[], None] | None = None,
        on_finish: Callable[[], None] | None = None,
        preemptible: bool = False,
    ):
        self.kind = kind
        self.preemptible = preemptible
        self.motion = motion
        self.max_speed = max_speed
        self.tolerance = tolerance
        self.settle_timeout = settle_timeout
        self.on_play = on_play
        self.on_finish = on_finish
        self.playing = False

    def start(self, measured: np.ndarray) -> None:
        self.origin = measured.copy()
        self.goal = self.motion.samples[-1]
        first = self.motion.samples[0]
        self.lead_in = min_jerk_duration(self.origin, first, self.max_speed)
        if self.lead_in > 0:
            self.lead_in = max(self.lead_in, 0.3)
        self.end = self.lead_in + self.motion.duration

    def step(self, measured: np.ndarray, t: float) -> np.ndarray | None:
        samples = self.motion.samples
        if t < self.lead_in:
            s = t / self.lead_in
            return self.origin + (10 * s**3 - 15 * s**4 + 6 * s**5) * (samples[0] - self.origin)
        if not self.playing:
            self.playing = True
            if self.on_play is not None:
                self.on_play()
        if t <= self.end:
            x = (t - self.lead_in) * self.motion.fps
            i = min(int(x), len(samples) - 2)
            return samples[i] + (x - i) * (samples[i + 1] - samples[i])
        settled = np.all(np.abs(self.goal - measured) <= self.tolerance)
        if settled or t > self.end + self.settle_timeout:
            return None
        return self.goal

    def progress(self, t: float) -> float:
        return min(t / self.end, 1.0) if self.end > 0 else 1.0

    def outcome(self, measured: np.ndarray) -> BehaviorOutcome:
        error = self.goal - measured
        reached = bool(np.all(np.abs(error) <= self.tolerance))
        notes = list(self.motion.notes)
        if not reached:
            notes.append("final pose not reached within tolerance")
        return BehaviorOutcome(
            reached=reached,
            final_error={f"{name}.pos": round(float(e), 2) for name, e in zip(JOINTS, error)},
            notes=notes,
        )

    def finish(self) -> None:
        if self.on_finish is not None:
            self.on_finish()
