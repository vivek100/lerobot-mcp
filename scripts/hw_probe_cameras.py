"""Read-only probe of the arm camera only, selected by device name (never opens other cameras).

Usage (PowerShell): uv run python scripts/hw_probe_cameras.py [--name Innomaker]
Outputs: artifacts/hw/arm_camera_<W>x<H>.jpg and a printed report per resolution.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from cv2_enumerate_cameras import enumerate_cameras

OUT = Path("artifacts/hw")
RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]


def find_index(name: str, backend: int) -> tuple[int, str]:
    matches = [c for c in enumerate_cameras(backend) if name.lower() in c.name.lower()]
    if not matches:
        names = [c.name for c in enumerate_cameras(backend)]
        raise SystemExit(f"No camera matching {name!r}; available: {names}")
    return matches[0].index, matches[0].name


def probe(index: int, backend: int, width: int, height: int) -> dict:
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("camera failed to open")
    for _ in range(10):  # let auto-exposure settle
        ok, frame = cap.read()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 1.5:
        ok, frame = cap.read()
        n += int(ok)
    fps = n / (time.perf_counter() - t0)
    cap.release()
    path = OUT / f"arm_camera_{frame.shape[1]}x{frame.shape[0]}.jpg"
    cv2.imwrite(str(path), frame)
    return {"requested": f"{width}x{height}", "got": f"{frame.shape[1]}x{frame.shape[0]}", "fps": round(fps, 1), "frame": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Innomaker")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    backend = cv2.CAP_DSHOW
    index, name = find_index(args.name, backend)
    print(f"arm camera: {name!r} -> DirectShow index {index}")
    for w, h in RESOLUTIONS:
        print(probe(index, backend, w, h))


if __name__ == "__main__":
    main()
