"""Verify motor ID -> joint name mapping by hand, read-only (interactive; run in your own terminal).

For each joint the script asks you to move ONLY that joint, samples every motor in the background
while you do, and reports which motor ID actually moved. Torque stays off and nothing is written,
except releasing torque (after you confirm you are holding the arm) if it is found on.

Usage (PowerShell): uv run python scripts/hw_identify_joints.py --port COM3 --id follower_01
"""

from __future__ import annotations

import argparse
import threading
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOW_TO_MOVE = {
    "shoulder_pan": "rotate the whole arm left/right at the base",
    "shoulder_lift": "tilt the upper arm up/down at the shoulder",
    "elbow_flex": "bend the elbow (forearm relative to upper arm)",
    "wrist_flex": "bend the wrist up/down",
    "wrist_roll": "twist the gripper around the forearm axis",
    "gripper": "open/close the gripper jaw",
}
MIN_TICKS = 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--id", default="follower_01")
    args = parser.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    bus = robot.bus
    bus.connect(handshake=True)
    results = []
    try:
        ids = {name: m.id for name, m in bus.motors.items()}
        torque = bus.sync_read("Torque_Enable", JOINTS, normalize=False)
        if any(torque.values()):
            input("Torque is ON. Hold the arm with one hand, then press ENTER to release torque...")
            bus.disable_torque()
        print("Torque is OFF. Move ONLY the joint asked for, then press ENTER.\n")

        for joint in JOINTS:
            start = bus.sync_read("Present_Position", JOINTS, normalize=False)
            travel = {m: 0 for m in JOINTS}
            done = threading.Event()

            def sample() -> None:
                while not done.is_set():
                    pos = bus.sync_read("Present_Position", JOINTS, normalize=False, num_retry=2)
                    for m in JOINTS:
                        travel[m] = max(travel[m], abs(pos[m] - start[m]))
                    time.sleep(0.02)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            input(f"[{joint}] {HOW_TO_MOVE[joint]} back and forth, then press ENTER...")
            done.set()
            sampler.join()

            moved = max(travel, key=travel.get)
            ok = moved == joint and travel[joint] >= MIN_TICKS
            others = {m: t for m, t in travel.items() if m != moved and t >= MIN_TICKS}
            verdict = "OK " if ok else "BAD"
            print(
                f"  {verdict} expected ID {ids[joint]} ({joint}); most movement: ID {ids[moved]} ({moved}) "
                f"{travel[moved]} ticks (~{travel[moved] * 360 / 4096:.0f} deg)"
            )
            if others:
                print(f"      also moved: {others} (fine if you bumped them, otherwise re-check)")
            results.append((joint, ok))
    finally:
        bus.port_handler.closePort()  # not bus.disconnect(): no torque writes

    bad = [j for j, ok in results if not ok]
    print("\nAll six joints map to the expected motor IDs." if not bad else f"\nMISMATCH or not moved: {bad}")


if __name__ == "__main__":
    main()
