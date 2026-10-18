"""Grab a frame from the laptop's integrated webcam (selected by name) to watch the arm from outside.

Usage (PowerShell): uv run python scripts/hw_laptop_cam.py --out artifacts/hw/laptop_cam.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from cv2_enumerate_cameras import enumerate_cameras

DEFAULT_NAME = "Integrated Webcam"


def open_camera(name: str = DEFAULT_NAME, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    matches = [c for c in enumerate_cameras(cv2.CAP_DSHOW) if name.lower() in c.name.lower()]
    if not matches:
        raise SystemExit(f"No camera named like {name!r}")
    cap = cv2.VideoCapture(matches[0].index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(15):  # let auto-exposure settle
        cap.read()
    return cap


def snapshot(cap: cv2.VideoCapture, out: Path) -> Path:
    for _ in range(3):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera returned no frame")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--out", default="artifacts/hw/laptop_cam.jpg")
    args = parser.parse_args()
    cap = open_camera(args.name)
    try:
        print(snapshot(cap, Path(args.out)))
    finally:
        cap.release()


if __name__ == "__main__":
    main()
