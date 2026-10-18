"""Render every animation as a strip of frames and report its lowest point above the table.

    uv run python scripts/render_animations.py            # all, sim speed budget
    uv run python scripts/render_animations.py --real     # fitted to the real arm's budget
    uv run python scripts/render_animations.py wave nod

Writes artifacts/animations/<name>.png and prints duration, fitting notes and table clearance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from so101_mcp.kinematics import ArmKinematics
from so101_mcp.runtime.animation import LIBRARY, fit_motion
from so101_mcp.runtime.safety import SpeedLimits

SIM_LIMITS = (np.array([-110, -100, -96.8, -95, -157, 0.0]), np.array([110, 100, 96.8, 95, 163, 100.0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args()

    if args.real:
        from so101_mcp.backends.real import REAL_SPEED as speed
    else:
        speed = SpeedLimits()
    kin = ArmKinematics()
    renderer = mujoco.Renderer(kin.model, 240, 240)
    out_dir = Path("artifacts/animations")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.names or list(LIBRARY):
        anim = LIBRARY[name]
        duration = anim.cycle_s
        motion = fit_motion(anim.base, anim.make(duration), duration, 30, *SIM_LIMITS,
                            speed.velocity(), speed.acceleration())
        lows = [kin.lowest_point(pose) for pose in motion.samples[::3]]
        low, link = min(lows, key=lambda x: x[0])
        print(f"{name:12s} {motion.duration:5.1f}s lowest {low * 100:5.1f} cm ({link}) {'; '.join(motion.notes)}")

        tiles = []
        for i in np.linspace(0, len(motion.samples) - 1, args.frames).astype(int):
            pose = motion.samples[i]
            kin.set_pose(pose)
            row = []
            for azimuth in (90, 180):
                cam = mujoco.MjvCamera()
                cam.lookat[:] = [0.1, 0, 0.15]
                cam.distance, cam.elevation, cam.azimuth = 0.6, -5, azimuth
                renderer.update_scene(kin.data, camera=cam)
                row.append(Image.fromarray(renderer.render().copy()))
            tile = Image.new("RGB", (240, 480))
            tile.paste(row[0], (0, 0))
            tile.paste(row[1], (0, 240))
            ImageDraw.Draw(tile).text((4, 4), f"t={i / 30:.1f}s", fill=(255, 0, 0))
            tiles.append(tile)
        sheet = Image.new("RGB", (240 * len(tiles), 480))
        for k, tile in enumerate(tiles):
            sheet.paste(tile, (240 * k, 0))
        ImageDraw.Draw(sheet).text((4, 466), name, fill=(255, 0, 0))
        sheet.save(out_dir / f"{name}{'_real' if args.real else ''}.png")


if __name__ == "__main__":
    main()
