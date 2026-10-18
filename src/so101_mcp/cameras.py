"""USB cameras opened by name (DirectShow), read in a background thread that keeps only the latest frame."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

LAPTOP_WEBCAM = "Integrated Webcam"


class FrameGrabber:
    """Opens one camera by name and keeps only the latest RGB frame."""

    def __init__(self, name: str, width: int = 640, height: int = 480, fps: int = 30):
        from cv2_enumerate_cameras import enumerate_cameras

        matches = [c for c in enumerate_cameras(cv2.CAP_DSHOW) if name.lower() in c.name.lower()]
        if not matches:
            raise RuntimeError(f"No camera named like {name!r}")
        self.cap = cv2.VideoCapture(matches[0].index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._frame: np.ndarray | None = None
        self._last_frame_at: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera-{name}", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while self._frame is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._frame is None:
            self.close()
            raise RuntimeError(f"Camera {name!r} produced no frames")

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._last_frame_at = time.monotonic()
            else:
                time.sleep(0.01)

    def latest(self) -> np.ndarray:
        with self._lock:
            if self._frame is None or self._last_frame_at is None or time.monotonic() - self._last_frame_at > 1.0:
                raise RuntimeError("Camera frame missing or stale for over 1 second")
            return self._frame.copy()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2)
        self.cap.release()
