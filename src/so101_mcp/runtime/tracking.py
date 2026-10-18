"""Visual tracking: keep a detected target centred in a camera image, or turn toward it.

Wrist camera (eye-in-hand): the tracker first nudges ``shoulder_pan`` and ``wrist_flex`` a few degrees
and measures how the target moves in the image (a 2x2 image Jacobian), so it works with any camera
mounting (the sim's wrist camera is rotated 90 degrees). It then steers with the Jacobian's inverse.

Fixed camera (eye-to-hand): moving the arm does not move the target in the image, so the target's
horizontal image position maps directly to a pan angle (image centre = pan 0).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from so101_mcp.runtime.behaviors import Behavior, BehaviorOutcome

PAN, WFLEX = 0, 3
YUNET_REPO = "opencv/face_detection_yunet"
YUNET_REVISION = "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
DETECT_WIDTH = 320


@dataclass(frozen=True)
class Detection:
    x: float  # -1 (left edge) .. 1 (right edge)
    y: float  # -1 (top) .. 1 (bottom)
    area: float  # fraction of the image
    score: float
    bbox: tuple[float, float, float, float] | None = None  # normalized left, top, right, bottom


def _small(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= DETECT_WIDTH:
        return frame
    return cv2.resize(frame, (DETECT_WIDTH, round(h * DETECT_WIDTH / w)), interpolation=cv2.INTER_AREA)


class ColorDetector:
    """Largest blob of a saturated colour (HSV threshold + connected components, ~1 ms)."""

    HUES = {"red": [(0, 8), (172, 180)], "yellow": [(20, 35)], "green": [(45, 85)], "blue": [(100, 130)]}

    def __init__(self, color: str, min_area: float = 0.0005):
        if color not in self.HUES:
            raise ValueError(f"Unknown color {color!r}; choose from {list(self.HUES)}")
        self.color = color
        self.min_area = min_area

    def __call__(self, frame: np.ndarray) -> Detection | None:
        small = _small(frame)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.HUES[self.color]:
            mask |= cv2.inRange(hsv, (lo, 120, 60), (hi, 255, 255))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return None
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        h, w = mask.shape
        area = stats[i, cv2.CC_STAT_AREA] / (w * h)
        if area < self.min_area:
            return None
        cx, cy = centroids[i]
        bx, by, bw, bh = stats[i, :4]
        return Detection(cx / w * 2 - 1, cy / h * 2 - 1, float(area), 1.0,
                         (float(bx / w), float(by / h), float((bx + bw) / w), float((by + bh) / h)))


def yunet_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(YUNET_REPO, YUNET_FILE, revision=YUNET_REVISION)


class FaceDetector:
    """Largest face from OpenCV's YuNet (CPU, a few ms at 320 px)."""

    def __init__(self, model_path: str, score: float = 0.75):
        self.detector = cv2.FaceDetectorYN.create(model_path, "", (DETECT_WIDTH, 240), score)

    def __call__(self, frame: np.ndarray) -> Detection | None:
        small = _small(frame)
        h, w = small.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        if faces is None or len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
        score = float(max(faces, key=lambda f: f[2] * f[3])[-1])
        bbox = tuple(float(v) for v in np.clip([x / w, y / h, (x + fw) / w, (y + fh) / h], 0, 1))
        return Detection(float((x + fw / 2) / w * 2 - 1), float((y + fh / 2) / h * 2 - 1), float(fw * fh / (w * h)), score, bbox)


class TrackingBehavior(Behavior):
    """Follow a target for ``seconds``; ends early if the target stays lost for ``lost_timeout_s``."""

    kind = "track"

    def __init__(
        self,
        frame_source: Callable[[], np.ndarray],
        detector: Callable[[np.ndarray], Detection | None],
        mode: Literal["servo", "point"],
        seconds: float,
        lower: np.ndarray,
        upper: np.ndarray,
        fps: int,
        point_sign: float = 1.0,
        point_span_deg: float = 40.0,
        detect_every: int = 2,
        lost_timeout_s: float = 3.0,
        probe_deg: float = 5.0,
        gain: float = 0.35,
        max_step_deg: float = 3.0,
        deadband: float = 0.04,
        on_finish: Callable[[], None] | None = None,
    ):
        self.frame_source = frame_source
        self.detector = detector
        self.mode = mode
        self.seconds = seconds
        self.lower, self.upper = lower, upper
        self.fps = fps
        self.point_sign = point_sign
        self.point_span = point_span_deg
        self.detect_every = detect_every
        self.lost_timeout = lost_timeout_s
        self.probe_deg = probe_deg
        self.gain = gain
        self.max_step = max_step_deg
        self.deadband = deadband
        self.on_finish = on_finish

    # ---------- lifecycle ----------
    def start(self, measured: np.ndarray) -> None:
        self.origin = measured.copy()
        self.target = measured.copy()
        self.ticks = 0
        self.last: Detection | None = None
        self.last_seen_t: float | None = None
        self.detections = 0
        self.frames = 0
        self.centered = 0
        self.ended: str | None = None
        self.jacobian: np.ndarray | None = None
        self.inverse: np.ndarray | None = None
        self.track_t0: float | None = None
        self.notes: list[str] = []
        self.probe = self._probe_plan() if self.mode == "servo" else []
        self.probe_samples: dict[str, list[np.ndarray]] = {}
        if self.mode == "point":
            self.track_t0 = 0.0

    def _probe_plan(self) -> list[tuple[float, np.ndarray, str | None]]:
        """Segments (end time, offset from origin, measurement key). Nudges go toward each joint's mid-range."""
        offsets = []
        for joint in (PAN, WFLEX):
            mid = (self.lower[joint] + self.upper[joint]) / 2
            sign = 1.0 if self.origin[joint] <= mid else -1.0
            offset = np.zeros_like(self.origin)
            offset[joint] = sign * self.probe_deg
            offsets.append(offset)
        zero = np.zeros_like(self.origin)
        plan, t = [], 0.0
        for duration, offset, key in [
            (0.6, zero, "base"), (0.8, offsets[0], "pan"), (0.5, zero, None),
            (0.8, offsets[1], "wrist"), (0.5, zero, None),
        ]:
            t += duration
            plan.append((t, offset, key))
        self.probe_offsets = offsets
        return plan

    def step(self, measured: np.ndarray, t: float) -> np.ndarray | None:
        if self.ended is not None or t >= self.seconds:
            return None
        detection = self._detect() if self.ticks % self.detect_every == 0 else None
        self.ticks += 1
        if detection is not None:
            self.last, self.last_seen_t = detection, t
            self.detections += 1
        if self.track_t0 is None:
            return self._probe_step(t, detection)
        if t - max(self.last_seen_t or 0.0, self.track_t0) > self.lost_timeout:
            self.ended = f"target lost for {self.lost_timeout:.0f} s"
            return None
        if detection is not None:
            self._update(detection)
        return self.target

    def _detect(self) -> Detection | None:
        self.frames += 1
        return self.detector(self.frame_source())

    def _probe_step(self, t: float, detection: Detection | None) -> np.ndarray | None:
        for end, offset, key in self.probe:
            if t < end:
                if key is not None and detection is not None and t > end - 0.25:
                    self.probe_samples.setdefault(key, []).append(np.array([detection.x, detection.y]))
                return self.origin + offset
        missing = [k for k in ("base", "pan", "wrist") if not self.probe_samples.get(k)]
        if missing:
            self.ended = "target not visible" if "base" in missing else "target lost while calibrating camera axes"
            return None
        base = np.mean(self.probe_samples["base"], axis=0)
        columns = [
            (np.mean(self.probe_samples[key], axis=0) - base) / offset[joint]
            for key, offset, joint in (("pan", self.probe_offsets[0], PAN), ("wrist", self.probe_offsets[1], WFLEX))
        ]
        self.jacobian = np.column_stack(columns)  # d(image x, y) / d(pan, wrist_flex) per degree
        if abs(np.linalg.det(self.jacobian)) < 1e-5 or np.linalg.cond(self.jacobian) > 25:
            self.ended = (
                "the target barely moves in this camera when the arm moves (fixed camera, or target too far); "
                "use a fixed-camera mode or move closer"
            )
            return None
        self.inverse = np.linalg.inv(self.jacobian)
        self.notes.append(f"image Jacobian per degree (x, y by pan, wrist_flex): {np.round(self.jacobian, 4).tolist()}")
        self.track_t0 = t
        return self.target

    def _update(self, detection: Detection) -> None:
        error = np.array([detection.x, detection.y])
        if self.mode == "servo" and np.max(np.abs(error)) < 0.15:
            self.centered += 1
        if self.mode == "point":
            # A fixed camera's image x is an absolute bearing: map it around pan 0 (facing the camera's line).
            goal = self.point_sign * self.point_span * detection.x
            self.target[PAN] += 0.3 * (goal - self.target[PAN])
        elif np.max(np.abs(error)) > self.deadband:
            delta = np.clip(-self.gain * self.inverse @ error, -self.max_step, self.max_step)
            self.target[PAN] += delta[0]
            self.target[WFLEX] += delta[1]
        self.target = np.clip(self.target, self.lower, self.upper)

    def progress(self, t: float) -> float:
        return min(t / self.seconds, 1.0)

    def metrics(self) -> dict[str, float]:
        out = {
            "calibrated": float(self.track_t0 is not None),
            "detections": float(self.detections),
            "frames": float(self.frames),
            "image_centering_applicable": float(self.mode == "servo"),
        }
        if self.mode == "servo":
            out["centered_detections"] = float(self.centered)
        if self.last is not None:
            out.update(error_x=round(self.last.x, 3), error_y=round(self.last.y, 3), target_area=round(self.last.area, 4))
        return out

    def outcome(self, measured: np.ndarray) -> BehaviorOutcome:
        notes = list(self.notes)
        if self.detections:
            centred = f", centred in {self.centered}" if self.mode == "servo" else ""
            notes.append(f"target seen in {self.detections}/{self.frames} frames{centred}")
        on_target = False
        if self.last is not None:
            if self.mode == "point":
                bearing_goal = self.point_sign * self.point_span * self.last.x
                on_target = abs(float(measured[PAN]) - bearing_goal) < 3.0
                notes.append("fixed camera: reached means pan within 3 degrees of the approximate bearing, not image centering")
            else:
                on_target = max(abs(self.last.x), abs(self.last.y)) < 0.2
        return BehaviorOutcome(reached=bool(self.ended is None and on_target), notes=notes, error=self.ended)

    def finish(self) -> None:
        if self.on_finish is not None:
            self.on_finish()
