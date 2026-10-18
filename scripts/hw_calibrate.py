"""Calibrate the SO-101 follower with torque OFF the whole time (interactive; run in your own terminal).

Why not plain `lerobot-calibrate`: it calls `connect()`, whose `configure()` re-enables torque before
calibrating. On this arm every motor's Goal_Position register read 0 while the joints sat near 2048,
so enabling torque could swing each joint ~180 degrees at full speed. This script runs LeRobot's own
`calibrate()` routine without ever enabling torque.

Never write Goal_Position here: on STS3215 servos a goal write turns torque ON. Goals are aligned to
the present position only by the real-arm backend, immediately before it deliberately enables torque.

Usage (PowerShell): uv run python scripts/hw_calibrate.py --port COM3 --id follower_01
"""

from __future__ import annotations

import argparse
import json

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FULL_TURN = "wrist_roll"
# A recorded edge this close to 0/4095 means the encoder wrapped: the middle pose was off-center.
WRAP_MARGIN = 30


def release_torque_if_on(bus) -> None:
    torque = bus.sync_read("Torque_Enable", JOINTS, normalize=False)
    on = [m for m, v in torque.items() if v]
    if not on:
        return
    print(f"\nTorque is ON for {on}: the arm is holding its pose and will go limp when released.")
    input("Hold the arm with one hand (support the upper arm and forearm), then press ENTER to release torque...")
    bus.disable_torque()
    print("Torque released.")


def wrapped_joints(data: dict) -> list[str]:
    return [
        joint
        for joint, cal in data.items()
        if joint != FULL_TURN and (cal["range_min"] <= WRAP_MARGIN or cal["range_max"] >= 4095 - WRAP_MARGIN)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--id", default="follower_01")
    args = parser.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    bus = robot.bus
    bus.connect(handshake=True)
    try:
        release_torque_if_on(bus)
        # An existing file makes calibrate() ask to reuse it; ENTER would keep the old (bad) ranges.
        robot.calibration = {}
        print(
            "\nTorque is OFF. Nothing on this arm will move by itself during calibration.\n"
            "Step 1: Hold the arm like the calibration video: upper arm and forearm roughly\n"
            "        horizontal and straight out, wrist straight, gripper half open. Press ENTER.\n"
            "Step 2: Move each joint (except wrist_roll) slowly through its FULL range, end to end.\n"
            "        Watch the table: shoulder_lift and elbow_flex must NOT reach 0 or 4095. Press ENTER.\n"
        )
        robot.calibrate()
    finally:
        bus.port_handler.closePort()  # not bus.disconnect(): no torque writes at all

    data = json.loads(robot.calibration_fpath.read_text())
    print(f"\nSaved {robot.calibration_fpath}")
    for joint, cal in data.items():
        print(f"  {joint:14s} homing_offset={cal['homing_offset']:6d} range=[{cal['range_min']}, {cal['range_max']}]")
    bad = wrapped_joints(data)
    if bad:
        print(
            f"\nBAD CALIBRATION: {bad} reached the 0/4095 encoder edge, so their range wrapped around.\n"
            "Their Step-1 pose was too far from the middle. Re-run and center those joints more carefully."
        )
        raise SystemExit(1)
    print("\nRanges look valid.")


if __name__ == "__main__":
    main()
