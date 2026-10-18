"""Strictly read-only probe of the SO-101 follower motor bus. Writes nothing to any motor.

Opens the port without LeRobot's handshake, pings the bus, reads raw registers per motor, and closes
the port WITHOUT touching torque (LeRobot's default disconnect writes Torque_Enable=0; a raised arm
would drop if torque were on).

Usage (PowerShell): uv run python scripts/hw_probe_motors.py --port COM3
"""

from __future__ import annotations

import argparse
import json
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

MOTORS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
REGISTERS = [
    "Torque_Enable",
    "Present_Position",
    "Present_Load",
    "Present_Current",
    "Present_Voltage",
    "Present_Temperature",
    "Status",
    "Min_Position_Limit",
    "Max_Position_Limit",
    "Homing_Offset",
    "Torque_Limit",
    "Max_Torque_Limit",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--samples", type=int, default=20, help="position samples for the noise check")
    args = parser.parse_args()

    bus = FeetechMotorsBus(
        port=args.port,
        motors={name: Motor(mid, "sts3215", MotorNormMode.DEGREES) for name, mid in MOTORS.items()},
    )
    bus.connect(handshake=False)
    report: dict = {"port": args.port, "baudrate": bus.get_baudrate() if hasattr(bus, "get_baudrate") else None}
    try:
        found = bus.broadcast_ping(num_retry=2) or {}
        report["broadcast_ping"] = {str(k): v for k, v in sorted(found.items())}
        missing = [name for name, mid in MOTORS.items() if mid not in found]
        report["missing_motors"] = missing

        per_motor: dict = {}
        for name, mid in MOTORS.items():
            if mid not in found:
                continue
            regs = {}
            for reg in REGISTERS:
                try:
                    regs[reg] = bus.read(reg, name, normalize=False)
                except Exception as exc:  # noqa: BLE001
                    regs[reg] = f"ERR {exc.__class__.__name__}"
            per_motor[name] = regs
        report["motors"] = per_motor

        present = [n for n in MOTORS if MOTORS[n] in found]
        if present:
            samples = []
            t0 = time.perf_counter()
            for _ in range(args.samples):
                samples.append(bus.sync_read("Present_Position", present, normalize=False))
            dt = (time.perf_counter() - t0) / args.samples
            report["sync_read_ms"] = round(1000 * dt, 2)
            report["position_jitter_ticks"] = {
                n: max(s[n] for s in samples) - min(s[n] for s in samples) for n in present
            }
    finally:
        bus.port_handler.closePort()  # deliberately not bus.disconnect(): that writes Torque_Enable=0

    cal_dir = HF_LEROBOT_CALIBRATION / "robots" / "so101_follower"
    report["calibration_files"] = sorted(p.name for p in cal_dir.glob("*.json")) if cal_dir.exists() else []
    report["calibration_dir"] = str(cal_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
