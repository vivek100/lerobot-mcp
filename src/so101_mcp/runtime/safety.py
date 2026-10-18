"""Last line of software safety between any behaviour and the backend.

Every target passes through ``SafetyLimiter.apply`` on the control thread: non-finite values are
rejected, targets are clamped to soft joint limits, the change from the previously commanded
target is capped per tick (speed limit), and the table guard refuses steps that would bring the arm
into the table. Physical e-stop remains the real safety authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101_mcp.backends.base import JointLimits
from so101_mcp.joints import GRIPPER_INDEX


@dataclass(frozen=True)
class SpeedLimits:
    """Per-tick speed caps enforced by the limiter, plus acceleration budgets animations are fitted to.

    ``arm_deg_per_s`` caps requested jog speeds and is the default cap for every arm joint;
    ``joint_deg_per_s`` overrides it per arm joint (pan, lift, elbow, wrist flex, wrist roll).
    ``accel`` (pan, lift, elbow, wrist flex, wrist roll, gripper; deg/s^2 or units/s^2) is not
    enforced per tick: animations are scaled to stay inside it so the servos can follow them.
    """

    arm_deg_per_s: float = 90.0
    gripper_units_per_s: float = 150.0
    joint_deg_per_s: tuple[float, float, float, float, float] | None = None
    accel: tuple[float, float, float, float, float, float] | None = None

    def velocity(self) -> np.ndarray:
        arm = self.joint_deg_per_s or (self.arm_deg_per_s,) * GRIPPER_INDEX
        return np.array([*arm, self.gripper_units_per_s], dtype=np.float64)

    def acceleration(self) -> np.ndarray | None:
        return None if self.accel is None else np.array(self.accel, dtype=np.float64)


class UnsafeTargetError(ValueError):
    pass


class TableGuardError(UnsafeTargetError):
    pass


# Agent Cartesian moves and the runtime table guard share this floor. 0.3 cm is tight enough to
# close on a marker lying on the table without letting a mesh face slam the top.
TABLE_CLEARANCE_M = 0.003


class TableGuard:
    """Refuses any per-tick target that brings a moving link below the table top plus a clearance.

    The allowed floor is the clearance, lowered only to the measured pose's height when the arm starts
    (or is resynced) already low, e.g. folded on the table. The floor then ratchets up as the arm rises
    and never comes back down, so a slow descent cannot creep below it one small step at a time.
    """

    def __init__(
        self,
        rest_pose: np.ndarray | None = None,
        clearance_m: float = TABLE_CLEARANCE_M,
        tolerance_m: float = 0.002,
        dock_deg: float = 10.0,
        table_top_z: float | None = None,
    ):
        from so101_mcp.kinematics import TABLE_TOP_Z, ArmKinematics

        self.kin = ArmKinematics(table_top_z=TABLE_TOP_Z if table_top_z is None else table_top_z)
        self.clearance = clearance_m
        self.tolerance = tolerance_m
        # The rest pose is where the arm physically sits on the table; the last few degrees of the
        # approach to it are allowed even though they are below the clearance.
        self.rest_pose = None if rest_pose is None else np.asarray(rest_pose, dtype=np.float64)
        self.dock_deg = dock_deg
        self.default_clearance = clearance_m
        self.floor = clearance_m

    def set_clearance(self, clearance_m: float | None) -> None:
        """Per-job clearance (None restores the default). The floor never rises above the clearance."""
        self.clearance = self.default_clearance if clearance_m is None else clearance_m
        self.floor = min(self.floor, self.clearance)

    def reset(self, measured: np.ndarray) -> None:
        low, _ = self.kin.lowest_point(measured)
        self.floor = min(self.clearance, low - self.tolerance)

    def check(self, target: np.ndarray) -> None:
        low, link = self.kin.lowest_point(target)
        if self.rest_pose is not None and np.all(np.abs(target[:5] - self.rest_pose[:5]) <= self.dock_deg):
            return
        if low >= self.floor:
            self.floor = min(self.clearance, max(self.floor, low - self.tolerance))
            return
        raise TableGuardError(
            f"table guard: this move would bring the {link} to {low * 100:.1f} cm from the table top "
            f"(minimum {self.clearance * 100:.1f} cm); motion stopped"
        )


class SafetyLimiter:
    def __init__(
        self,
        limits: JointLimits,
        fps: int,
        speed: SpeedLimits = SpeedLimits(),
        table_guard: TableGuard | None = None,
    ):
        self.limits = limits
        self.fps = fps
        self.speed = speed
        self.table_guard = table_guard
        self.max_step = speed.velocity() / fps
        self.last_command: np.ndarray | None = None

    def clamp_to_limits(self, target: np.ndarray) -> tuple[np.ndarray, bool]:
        clamped = np.clip(target, self.limits.lower, self.limits.upper)
        return clamped, bool(np.any(clamped != target))

    def apply(self, target: np.ndarray, measured: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Return the target actually sent and any notes about how it was altered."""
        target = np.asarray(target, dtype=np.float64)
        if target.shape != self.limits.lower.shape or not np.all(np.isfinite(target)):
            raise UnsafeTargetError(f"Rejected non-finite or mis-shaped target {target.tolist()}")
        notes: list[str] = []
        target, clipped = self.clamp_to_limits(target)
        if clipped:
            notes.append("clamped_to_joint_limits")
        reference = self.last_command if self.last_command is not None else measured
        delta = np.clip(target - reference, -self.max_step, self.max_step)
        if np.any(delta != target - reference):
            notes.append("speed_limited")
        safe = reference + delta
        if self.table_guard is not None:
            try:
                self.table_guard.check(safe)
            except TableGuardError:
                # The arm stays where it was stopped; from there it may move anywhere not lower.
                self.table_guard.reset(reference)
                raise
        self.last_command = safe
        return safe, notes

    def resync(self, measured: np.ndarray) -> None:
        """Forget the commanded history, e.g. after a reset, so motion restarts from the measured pose."""
        self.last_command = np.asarray(measured, dtype=np.float64).copy()
        if self.table_guard is not None:
            self.table_guard.reset(self.last_command)
