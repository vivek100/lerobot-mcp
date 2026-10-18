"""Cartesian moves for agents: "3 cm forward, 2 cm down" becomes a checked joint-space path.

Coordinates are the table frame at the arm's base: forward = +x (away from the base, over the table),
left = +y (the arm's left), up = +z (table top is 0). The gripper point is the MuJoCo ``gripperframe``
site, between the jaw tips. Tilt is the finger axis angle below horizontal (90 = pointing straight
down, negative = pointing above horizontal); roll is the wrist_roll joint.

Every plan is checked before any motor moves: each waypoint must be solvable within the soft joint
limits and keep every moving link above the table clearance (same rule as the runtime's table guard).
The runtime still checks every tick while the path runs.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from so101_mcp.joints import JOINTS
from so101_mcp.kinematics import ArmKinematics
from so101_mcp.runtime.safety import TABLE_CLEARANCE_M

ARM = 4  # pan, lift, elbow, wrist_flex solve position + tilt; wrist_roll is set directly
ROLL = 4
TILT_WEIGHT = 0.1  # metres of position error that count as one radian of tilt error
STEP_M = 0.01
STEP_TILT_DEG = 5.0
STEP_ROLL_DEG = 10.0
MAX_JUMP_DEG = 15.0
ROOM_MAX_CM = 30.0
ROOM_TILT_MAX_DEG = 90.0


def _limiter(reason: str) -> str:
    """Short cause for room listings: 'table', 'reach', '<joint> limit', 'elbow/wrist flip'."""
    if "above the table" in reason:
        return "table"
    if "flip" in reason:
        return "elbow/wrist flip"
    if "(" in reason:
        return reason[reason.rindex("(") + 1 : reason.rindex(")")]
    return reason


@dataclass(frozen=True)
class GripperPose:
    forward_cm: float
    left_cm: float
    up_cm: float
    tilt_down_deg: float
    roll_deg: float
    opening: float
    table_clearance_cm: float
    lowest_link: str


@dataclass(frozen=True)
class Room:
    direction: str
    amount: float
    unit: str
    limited_by: str


@dataclass
class Plan:
    ok: bool
    reason: str
    waypoints: list[np.ndarray]
    target: np.ndarray  # xyz (m), tilt (rad), roll (deg)


class CartesianPlanner:
    """Thread-safe (one lock); owns its own kinematics model, separate from the runtime's guard."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray, clearance_m: float = TABLE_CLEARANCE_M, tolerance_m: float = 0.002):
        self.kin = ArmKinematics()
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.clearance = clearance_m
        self.tolerance = tolerance_m
        self._lock = threading.Lock()

    # ---------- forward ----------
    def _fk(self, joints: np.ndarray) -> tuple[np.ndarray, float]:
        """Gripper point (m) and tilt (rad) for a pose."""
        self.kin.set_pose(joints)
        d = self.kin.data
        point = d.site_xpos[self.kin.gripper_site].copy()
        axis = d.site_xmat[self.kin.gripper_site].reshape(3, 3)[:, 0]
        radial = np.array([point[0], point[1]])
        norm = np.linalg.norm(radial)
        radial = radial / norm if norm > 1e-6 else np.array([1.0, 0.0])
        return point, math.atan2(-axis[2], float(axis[:2] @ radial))

    def pose(self, joints: np.ndarray) -> GripperPose:
        with self._lock:
            point, tilt = self._fk(joints)
            low, link = self.kin.lowest_point(joints)
        return GripperPose(
            forward_cm=round(point[0] * 100, 1),
            left_cm=round(point[1] * 100, 1),
            up_cm=round(point[2] * 100, 1),
            tilt_down_deg=round(math.degrees(tilt), 1),
            roll_deg=round(float(joints[ROLL]), 1),
            opening=round(float(joints[5]), 1),
            table_clearance_cm=round(low * 100, 1),
            lowest_link=link,
        )

    # ---------- inverse ----------
    def _solve(self, xyz: np.ndarray, tilt: float, roll: float, seed: np.ndarray) -> tuple[np.ndarray | None, str]:
        """Damped least squares on pan/lift/elbow/wrist_flex; numeric Jacobian (cheap FK)."""
        q = seed.copy()
        q[ROLL] = roll
        if not self.lower[ROLL] <= roll <= self.upper[ROLL]:
            return None, "wrist_roll limit"
        h = 0.05  # degrees
        for _ in range(80):
            point, t = self._fk(q)
            error = np.concatenate([xyz - point, [TILT_WEIGHT * (tilt - t)]])
            if np.linalg.norm(error[:3]) < 0.0015 and abs(tilt - t) < math.radians(1.0):
                return q, ""
            jac = np.zeros((4, ARM))
            for j in range(ARM):
                probe = q.copy()
                probe[j] += h
                p2, t2 = self._fk(probe)
                jac[:3, j] = (p2 - point) / h
                jac[3, j] = TILT_WEIGHT * (t2 - t) / h
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(4), error)
            step = np.clip(step, -8.0, 8.0)
            q[:ARM] = np.clip(q[:ARM] + step, self.lower[:ARM], self.upper[:ARM])
        at_limit = [JOINTS[j] for j in range(ARM) if q[j] <= self.lower[j] + 0.5 or q[j] >= self.upper[j] - 0.5]
        return None, (f"{at_limit[0]} limit" if at_limit else "reach")

    def _floor(self, start: np.ndarray) -> float:
        low, _ = self.kin.lowest_point(start)
        return min(self.clearance, low - self.tolerance)

    # ---------- planning ----------
    def plan(self, joints: np.ndarray, forward_cm=0.0, left_cm=0.0, up_cm=0.0, tilt_down_deg=0.0, roll_deg=0.0) -> Plan:
        with self._lock:
            return self._plan(np.asarray(joints, dtype=np.float64), forward_cm, left_cm, up_cm, tilt_down_deg, roll_deg)

    def _plan(self, joints: np.ndarray, forward_cm: float, left_cm: float, up_cm: float, tilt_down_deg: float, roll_deg: float) -> Plan:
        point, tilt = self._fk(joints)
        delta = np.array([forward_cm, left_cm, up_cm]) / 100.0
        d_tilt = math.radians(tilt_down_deg)
        target = np.array([*(point + delta), tilt + d_tilt, joints[ROLL] + roll_deg])
        steps = max(
            1,
            math.ceil(np.linalg.norm(delta) / STEP_M),
            math.ceil(abs(tilt_down_deg) / STEP_TILT_DEG),
            math.ceil(abs(roll_deg) / STEP_ROLL_DEG),
        )
        floor = self._floor(joints)
        waypoints, previous = [], joints
        for i in range(1, steps + 1):
            s = i / steps
            q, why = self._solve(point + s * delta, tilt + s * d_tilt, joints[ROLL] + s * roll_deg, previous)
            where = f"{s * 100:.0f}% of the way"
            if q is None:
                return Plan(False, f"unreachable at {where} ({why})", [], target)
            if np.max(np.abs(q[:ARM] - previous[:ARM])) > MAX_JUMP_DEG:
                return Plan(False, f"the arm would have to flip its elbow or wrist at {where}", [], target)
            low, link = self.kin.lowest_point(q)
            if low < floor:
                return Plan(
                    False,
                    f"the {link} would go to {low * 100:.1f} cm above the table at {where} "
                    f"(minimum {self.clearance * 100:.1f} cm)",
                    [],
                    target,
                )
            floor = min(self.clearance, max(floor, low - self.tolerance))
            waypoints.append(q)
            previous = q
        return Plan(True, "", waypoints, target)

    def _path_ok(self, joints: np.ndarray, **move: float) -> tuple[bool, str]:
        plan = self._plan(joints, move.get("forward_cm", 0.0), move.get("left_cm", 0.0), move.get("up_cm", 0.0),
                          move.get("tilt_down_deg", 0.0), 0.0)
        return plan.ok, _limiter(plan.reason)

    def room(self, joints: np.ndarray) -> list[Room]:
        """How far `move` would accept going in each direction from here (the whole path is checked).

        Holds the current tilt and roll; changing tilt changes the reach.
        """
        joints = np.asarray(joints, dtype=np.float64)
        rooms: list[Room] = []
        with self._lock:
            axes = {
                "forward": ("forward_cm", 1.0), "back": ("forward_cm", -1.0),
                "left": ("left_cm", 1.0), "right": ("left_cm", -1.0),
                "up": ("up_cm", 1.0), "down": ("up_cm", -1.0),
            }
            for name, (key, sign) in axes.items():
                amount, why = self._search(lambda d: self._path_ok(joints, **{key: sign * d}), ROOM_MAX_CM)
                rooms.append(Room(name, amount, "cm", why))
            for name, sign in (("tilt_down", 1.0), ("tilt_up", -1.0)):
                amount, why = self._search(lambda d: self._path_ok(joints, tilt_down_deg=sign * d), ROOM_TILT_MAX_DEG)
                rooms.append(Room(name, amount, "deg", why))
        rooms.append(Room("roll_positive", round(float(self.upper[ROLL] - joints[ROLL]), 1), "deg", "wrist_roll limit"))
        rooms.append(Room("roll_negative", round(float(joints[ROLL] - self.lower[ROLL]), 1), "deg", "wrist_roll limit"))
        return rooms

    @staticmethod
    def _search(ok, maximum: float, iterations: int = 6) -> tuple[float, str]:
        good, why = ok(min(0.5, maximum))
        if not good:
            return 0.0, why
        good, why = ok(maximum)
        if good:
            return maximum, f"checked up to {maximum:g}"
        lo, hi, reason = 0.0, maximum, why
        for _ in range(iterations):
            mid = (lo + hi) / 2
            good, why = ok(mid)
            if good:
                lo = mid
            else:
                hi, reason = mid, why
        return round(lo, 1), reason
