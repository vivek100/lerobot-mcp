"""First powered test of the SO-101 follower: hold in place, then 5 degree nudges one joint at a time.

Interactive; run in your own terminal and watch the arm with a hand near the motor power switch.

Safety layers:
* Goal_Position is set to Present_Position immediately before torque is enabled (on STS3215 a goal
  write itself turns torque on), so "torque on" means "hold exactly here".
* Servo-side speed cap (Goal_Velocity) and gentle acceleration, set after the hold is verified.
* Each nudge moves AWAY from the joint's nearest calibrated end stop (into open space, never into the
  table or a stop), ramps over ~1 s, then returns to where it started.
* Load and tracking error are watched every step; past a threshold the joint is frozen where it is.
* On any error or Ctrl+C the arm holds its current pose (torque stays on, no drop). Torque is only
  released at the end after you confirm you are supporting the arm.

Usage (PowerShell): uv run python scripts/hw_first_motion.py --port COM3 --id follower_01
Unattended, with laptop-webcam frames before/after each nudge:
    uv run python scripts/hw_first_motion.py --auto --webcam
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

sys.path.insert(0, str(Path(__file__).parent))
from hw_laptop_cam import open_camera, snapshot  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Lightest / least gravity-loaded joints first; shoulder_lift carries the whole arm, so it goes last.
NUDGE_ORDER = ["shoulder_pan", "wrist_roll", "wrist_flex", "gripper", "elbow_flex", "shoulder_lift"]
TICKS_PER_DEG = 4096 / 360
NUDGE_DEG = 5.0
EDGE_MARGIN = 40  # ticks kept away from each calibrated range end
VELOCITY_CAP = 200  # steps/s on the servo itself (~18 deg/s)
ACCELERATION = 10  # x100 steps/s^2
RAMP_S = 1.0
RATE_HZ = 30
HOLD_DRIFT_MAX = 20  # ticks a joint may move when torque is first enabled
LOAD_STOP = 700  # of 1000 (signed Present_Load magnitude)
TRACK_STOP = 150  # ticks between commanded and measured position
TORQUE_LIMIT = 400  # of 1000; these are 12 V servos, cap force for the first test
REACHED_TOL = 15  # ticks
# Joints lifting the arm against gravity sag a little under the capped torque (P-control droop).
GRAVITY_TOL = {"elbow_flex": 35, "shoulder_lift": 35}
AUTO = False
OUT_DIR = Path("artifacts/hw/first_motion")


class Abort(RuntimeError):
    pass


class Webcam:
    def __init__(self, enabled: bool):
        self.cap = open_camera() if enabled else None
        if enabled:
            OUT_DIR.mkdir(parents=True, exist_ok=True)

    def shot(self, name: str) -> Path | None:
        return snapshot(self.cap, OUT_DIR / f"{name}.jpg") if self.cap else None

    def diff(self, a: Path | None, b: Path | None, name: str) -> None:
        if not (a and b):
            return
        fa, fb = cv2.imread(str(a)), cv2.imread(str(b))
        d = cv2.absdiff(fa, fb).max(axis=2)
        mask = (d > 35).astype(np.uint8) * 255
        overlay = fb.copy()
        overlay[mask > 0] = (0, 0, 255)
        cv2.imwrite(str(OUT_DIR / f"{name}.jpg"), np.hstack([fa, cv2.addWeighted(fb, 0.5, overlay, 0.5, 0)]))

    def close(self) -> None:
        if self.cap:
            self.cap.release()


def read(bus, name: str) -> dict[str, int]:
    return bus.sync_read(name, JOINTS, normalize=False, num_retry=3)


def hold_here(bus) -> dict[str, int]:
    present = read(bus, "Present_Position")
    bus.sync_write("Goal_Position", present, normalize=False)
    return present


def ask(prompt: str, auto_answer: str = "") -> str:
    if AUTO:
        print(f"{prompt}[auto: {auto_answer or 'ENTER'}]")
        return auto_answer
    answer = input(prompt).strip().lower()
    if answer == "q":
        raise Abort("Stopped by user")
    return answer


def move_joint(bus, joint: str, start: dict[str, int], target: int) -> dict[str, int]:
    """Ramp one joint from its start to target; every other joint keeps its start goal."""
    goal = dict(start)
    steps = max(1, int(RAMP_S * RATE_HZ))
    peak_load = 0
    for i in range(1, steps + 1):
        goal[joint] = round(start[joint] + (target - start[joint]) * i / steps)
        bus.sync_write("Goal_Position", {joint: goal[joint]}, normalize=False)
        time.sleep(1 / RATE_HZ)
        pos, load = read(bus, "Present_Position"), read(bus, "Present_Load")
        peak_load = max(peak_load, max(abs(v) for v in load.values()))
        worst = max(JOINTS, key=lambda m: abs(load[m]))
        if abs(load[worst]) > LOAD_STOP:
            hold_here(bus)
            raise Abort(f"Load on {worst} reached {load[worst]} (limit {LOAD_STOP}); froze in place")
        if abs(goal[joint] - pos[joint]) > TRACK_STOP:
            hold_here(bus)
            raise Abort(f"{joint} lagging {abs(goal[joint] - pos[joint])} ticks behind its goal; froze in place")
    time.sleep(0.4)
    pos = read(bus, "Present_Position")
    print(
        f"    -> target {target}, reached {pos[joint]} (error {pos[joint] - target:+d} ticks, "
        f"{(pos[joint] - target) / TICKS_PER_DEG:+.1f} deg), peak load {peak_load}"
    )
    return pos


def nudge_target(joint: str, current: int, cal) -> int:
    lo, hi = cal.range_min + EDGE_MARGIN, cal.range_max - EDGE_MARGIN
    if joint == "gripper":
        delta = round(0.05 * (cal.range_max - cal.range_min))
    else:
        delta = round(NUDGE_DEG * TICKS_PER_DEG)
    center = (cal.range_min + cal.range_max) / 2
    direction = 1 if current <= center else -1
    return int(min(hi, max(lo, current + direction * delta)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--id", default="follower_01")
    parser.add_argument("--auto", action="store_true", help="no prompts; stop if any joint fails to reach its target")
    parser.add_argument("--webcam", action="store_true", help="laptop webcam frames before/after each nudge")
    parser.add_argument("--joints", nargs="+", choices=NUDGE_ORDER, help="only nudge these joints")
    args = parser.parse_args()
    global AUTO
    AUTO = args.auto

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    if not robot.calibration:
        raise SystemExit(f"No calibration file at {robot.calibration_fpath}; run scripts/hw_calibrate.py first")
    cam = Webcam(args.webcam)
    bus = robot.bus
    bus.connect(handshake=True)
    torque_on = False
    finished = False
    try:
        if not bus.is_calibrated:
            raise Abort("Calibration stored in the motors does not match the calibration file; re-run hw_calibrate.py")
        volts = read(bus, "Present_Voltage")
        temps = read(bus, "Present_Temperature")
        print(f"Motors OK. Supply {min(volts.values()) / 10:.1f} V, temperatures {min(temps.values())}-{max(temps.values())} C.")
        if any(read(bus, "Torque_Enable").values()):
            print("Note: torque is already on; the hold step re-aligns the goal to the present pose first.")

        print(
            "\nBefore torque goes on:\n"
            "  * Put the arm in its folded rest pose, resting on the table, gripper clear of objects.\n"
            "  * Clear the area around the arm and keep a hand near the motor power switch.\n"
            "  * Type q + ENTER at any prompt to stop (the arm will hold where it is)."
        )
        ask("Press ENTER to enable torque and HOLD the current pose...")

        cam.shot("00_before_torque")
        start = hold_here(bus)
        bus.enable_torque()
        torque_on = True
        for m in JOINTS:
            bus.write("Torque_Limit", m, min(TORQUE_LIMIT, 500 if m == "gripper" else 1000), normalize=False)
            bus.write("Acceleration", m, ACCELERATION, normalize=False)
            bus.write("Goal_Velocity", m, VELOCITY_CAP, normalize=False)
        time.sleep(1.5)
        cam.shot("01_holding")
        after = read(bus, "Present_Position")
        drift = {m: after[m] - start[m] for m in JOINTS}
        load = read(bus, "Present_Load")
        print(f"\nHOLD: drift ticks {drift}")
        print(f"      load {load}")
        if max(abs(d) for d in drift.values()) > HOLD_DRIFT_MAX:
            hold_here(bus)
            raise Abort(f"A joint moved more than {HOLD_DRIFT_MAX} ticks when torque came on; stopping")
        ask("The arm should be stiff and NOT have moved. Correct? ENTER = yes, q = stop: ")

        results = {}
        order = [j for j in NUDGE_ORDER if not args.joints or j in args.joints]
        for n, joint in enumerate(order, start=2):
            cal = robot.calibration[joint]
            here = read(bus, "Present_Position")
            target = nudge_target(joint, here[joint], cal)
            amount = abs(target - here[joint]) / TICKS_PER_DEG
            unit = "% of gripper range" if joint == "gripper" else "deg"
            shown = 5.0 if joint == "gripper" else amount
            if ask(f"\n[{joint}] (ID {cal.id}) nudge {shown:.1f} {unit} and back. ENTER = go, s = skip, q = stop: ") == "s":
                results[joint] = "skipped"
                continue
            before = cam.shot(f"{n:02d}_{joint}_a_before")
            moved = move_joint(bus, joint, here, target)
            after_move = cam.shot(f"{n:02d}_{joint}_b_moved")
            cam.diff(before, after_move, f"{n:02d}_{joint}_c_diff")
            back = move_joint(bus, joint, moved, here[joint])
            others = {m: moved[m] - here[m] for m in JOINTS if m != joint and abs(moved[m] - here[m]) > REACHED_TOL}
            tol = GRAVITY_TOL.get(joint, REACHED_TOL)
            reached = abs(moved[joint] - target) <= tol and abs(back[joint] - here[joint]) <= tol
            print(f"    encoder: {'reached target and returned' if reached else 'DID NOT REACH TARGET'}"
                  + (f"; other joints shifted {others}" if others else ""))
            if AUTO and not reached:
                raise Abort(f"{joint} did not reach its 5 degree target; stopping before trying other joints")
            seen = ask(f"    Did ONLY the {joint} move a little and come back? ENTER = yes, n = no: ")
            if AUTO:
                results[joint] = "encoder ok (check webcam diff)" if not others else f"encoder ok, others shifted {others}"
            else:
                results[joint] = "ok" if seen != "n" else "WRONG JOINT / NOT SEEN"

        print("\nResults:")
        for joint, verdict in results.items():
            print(f"  {joint:14s} {verdict}")
        finished = True

        final = read(bus, "Present_Position")
        at_rest = all(abs(final[m] - start[m]) <= 2 * GRAVITY_TOL.get(m, REACHED_TOL) for m in JOINTS)
        cam.shot("99_end")
        release_default = "" if at_rest else "k"
        if ask("\nRelease torque? Support the arm first (it goes limp). ENTER = release, k = keep holding: ",
               release_default) != "k":
            bus.disable_torque()
            torque_on = False
            print("Torque released.")
    except (Abort, KeyboardInterrupt) as exc:
        print(f"\nSTOPPED: {exc or 'Ctrl+C'}")
    finally:
        if torque_on:
            try:
                hold_here(bus)
            except Exception:  # noqa: BLE001
                pass
            print(
                "Torque is still ON and the arm is holding its pose (it will not drop).\n"
                "To release: support the arm and switch the motor power off, or re-run this script."
            )
        bus.port_handler.closePort()  # not bus.disconnect(): it would drop torque on a raised arm
        cam.close()
    if not finished:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
