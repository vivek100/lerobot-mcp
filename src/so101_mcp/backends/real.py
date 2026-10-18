"""Real SO-101 follower backend (Feetech STS3215 bus through LeRobot's ``SO101Follower``).

Differences from ``SO101Follower.connect()``, all for safety on this arm:
* ``configure()`` is not called: it toggles torque with whatever stale value sits in Goal_Position.
  Instead Goal_Position is set to Present_Position (which on STS3215 also enables torque, so the arm
  just holds), then torque is enabled explicitly.
* Servo-side caps are applied: Torque_Limit, Goal_Velocity (speed) and Acceleration.
* ``disconnect()`` never drops a raised arm: torque is released only if the arm is near its rest pose,
  otherwise it keeps holding and the port is closed.

Cameras are opened by name (DirectShow) and read in background threads, so the 30 Hz control loop
never waits for a frame.
"""

from __future__ import annotations

import numpy as np

from so101_mcp.backends.base import JointLimits, RobotBackend
from so101_mcp.cameras import LAPTOP_WEBCAM, FrameGrabber
from so101_mcp.joints import GRIPPER_INDEX, JOINTS
from so101_mcp.runtime.safety import SpeedLimits

TICKS_PER_DEG = 4095 / 360
# Runtime budget on the real arm: the shoulder and elbow carry the arm's weight and stay slow; the
# wrist and gripper are light, so gestures (nod, wave, the talking "mouth") can be quick.
REAL_SPEED = SpeedLimits(
    arm_deg_per_s=25.0,
    gripper_units_per_s=150.0,
    joint_deg_per_s=(40.0, 25.0, 30.0, 90.0, 90.0),
    accel=(200.0, 120.0, 150.0, 900.0, 900.0, 2000.0),
)
# Servo-side Goal_Velocity / Acceleration sit this far above the runtime budget, so the servos follow
# commands without lagging (lag under load reads as contact) and only cap runaway targets.
SERVO_HEADROOM = 1.5
MAX_SERVO_TICKS_PER_S = 3000
MAX_SERVO_ACCELERATION = 254  # register units of 100 ticks/s^2
# Folded on the table as measured on this arm after calibration (degrees, gripper 0-100), with the
# shoulder and elbow pulled just inside their soft limits.
REST_POSE = np.array([7.6, -100.5, 92.5, 74.1, 1.9, 1.0])
REST_TOLERANCE_DEG = 8.0


class RealBackend(RobotBackend):
    mode = "real"
    name = "so101-follower"

    def __init__(
        self,
        port: str = "COM3",
        robot_id: str = "follower_01",
        cameras: dict[str, str] | None = None,
        torque_limits: dict[str, int] | None = None,
        speed: SpeedLimits = REAL_SPEED,
        limit_margin_deg: float = 4.0,
        contact_error_deg: float = 8.0,
        contact_load: int = 300,
        contact_ticks: int = 5,
        stall_ticks: int = 15,
        max_temperature_c: int = 60,
    ):
        self.port = port
        self.robot_id = robot_id
        self.camera_specs = cameras if cameras is not None else {"wrist": "Innomaker", "front": LAPTOP_WEBCAM}
        # Of 1000. The shoulder and elbow carry the arm's weight when it reaches out.
        self.torque_limits = torque_limits or {
            "shoulder_pan": 500, "shoulder_lift": 800, "elbow_flex": 700,
            "wrist_flex": 500, "wrist_roll": 500, "gripper": 500,
        }
        self.speed = speed
        self.limit_margin_deg = limit_margin_deg
        self.contact_error_deg = contact_error_deg
        self.contact_load = contact_load
        self.contact_ticks = contact_ticks
        self.stall_ticks = stall_ticks
        self.max_temperature_c = max_temperature_c
        self.fps = 30
        self.camera_names: tuple[str, ...] = ()
        self._grabbers: dict[str, FrameGrabber] = {}
        self._blocked = {j: 0 for j in JOINTS}
        self._stalled = {j: 0 for j in JOINTS}
        self._measured = np.zeros(len(JOINTS))
        self._target: np.ndarray | None = None
        self._tick = 0
        self._temps: dict[str, int] = {}
        self.robot = None
        self.bus = None
        self._torque_on = False

    # ---------- lifecycle ----------
    def connect(self) -> None:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.robot = SO101Follower(SO101FollowerConfig(port=self.port, id=self.robot_id))
        if not self.robot.calibration:
            raise RuntimeError(f"No calibration at {self.robot.calibration_fpath}; run scripts/hw_calibrate.py")
        self.bus = self.robot.bus
        self.bus.connect(handshake=True)
        if not self.bus.is_calibrated:
            self.bus.port_handler.closePort()
            raise RuntimeError("Motor calibration does not match the calibration file; run scripts/hw_calibrate.py")

        try:
            present = self.bus.sync_read("Present_Position", list(JOINTS), normalize=False, num_retry=3)
            self.bus.sync_write("Goal_Position", present, normalize=False)
            # A servo that just engaged torque occasionally misses one status packet.
            self.bus.enable_torque(num_retry=3)
            self._torque_on = True
            velocity, acceleration = self._servo_caps()
            for i, joint in enumerate(JOINTS):
                self.bus.write("Torque_Limit", joint, self.torque_limits[joint], normalize=False, num_retry=3)
                self.bus.write("Acceleration", joint, acceleration[i], normalize=False, num_retry=3)
                self.bus.write("Goal_Velocity", joint, velocity[i], normalize=False, num_retry=3)
        except BaseException:
            self.bus.port_handler.closePort()
            raise

        for cam_name, device in self.camera_specs.items():
            try:
                self._grabbers[cam_name] = FrameGrabber(device)
            except RuntimeError:
                continue
        self.camera_names = tuple(self._grabbers)

    def _servo_caps(self) -> tuple[list[int], list[int]]:
        """Goal_Velocity (ticks/s) and Acceleration (100 ticks/s^2) registers from the runtime budget."""
        ticks_per_unit = np.full(len(JOINTS), TICKS_PER_DEG)
        cal = self.robot.calibration[JOINTS[GRIPPER_INDEX]]
        ticks_per_unit[GRIPPER_INDEX] = (cal.range_max - cal.range_min) / 100
        velocity = self.speed.velocity() * ticks_per_unit * SERVO_HEADROOM
        accel = self.speed.acceleration()
        accel = accel * ticks_per_unit * SERVO_HEADROOM / 100 if accel is not None else np.full(len(JOINTS), 20)
        return (
            [int(min(v, MAX_SERVO_TICKS_PER_S)) for v in velocity],
            [int(min(np.ceil(a), MAX_SERVO_ACCELERATION)) for a in accel],
        )

    def disconnect(self) -> None:
        for grabber in self._grabbers.values():
            grabber.close()
        self._grabbers.clear()
        if self.bus is None or not self.bus.is_connected:
            return
        try:
            if self._torque_on and self.at_rest():
                self.bus.disable_torque()
                self._torque_on = False
            elif self._torque_on:
                present = self.bus.sync_read("Present_Position", list(JOINTS), normalize=False)
                self.bus.sync_write("Goal_Position", present, normalize=False)
        finally:
            self.bus.port_handler.closePort()

    # ---------- runtime interface ----------
    def joint_limits(self) -> JointLimits:
        lower, upper = np.zeros(len(JOINTS)), np.zeros(len(JOINTS))
        for i, joint in enumerate(JOINTS):
            cal = self.robot.calibration[joint]
            if i == GRIPPER_INDEX:
                lower[i], upper[i] = 0.0, 100.0
                continue
            mid = (cal.range_min + cal.range_max) / 2
            lower[i] = (cal.range_min - mid) / TICKS_PER_DEG + self.limit_margin_deg
            upper[i] = (cal.range_max - mid) / TICKS_PER_DEG - self.limit_margin_deg
        return JointLimits(lower, upper)

    def read_joints(self) -> np.ndarray:
        pos = self.bus.sync_read("Present_Position", list(JOINTS), num_retry=2)
        self._measured = np.array([pos[j] for j in JOINTS], dtype=np.float64)
        return self._measured.copy()

    def command(self, target: np.ndarray) -> None:
        self.bus.sync_write("Goal_Position", {j: float(v) for j, v in zip(JOINTS, target)})
        self._target = np.asarray(target, dtype=np.float64).copy()
        self._tick += 1

    def render(self, camera: str) -> np.ndarray:
        if camera not in self._grabbers:
            raise ValueError(f"Unknown camera {camera!r}; available: {list(self._grabbers)}")
        return self._grabbers[camera].latest()

    def reset(self, seed: int | None) -> None:
        raise RuntimeError("reset is only available in simulation")

    def contact_faults(self) -> list[str]:
        """Blocked: loaded and well behind its goal. Stalled: pinned at its torque limit.

        Load alone is not contact: holding the arm out against gravity loads the shoulder heavily.
        """
        load = self.bus.sync_read("Present_Load", list(JOINTS), normalize=False, num_retry=2)
        faults = []
        lag = np.abs(self._target - self._measured) if self._target is not None else np.zeros(len(JOINTS))
        for i, joint in enumerate(JOINTS):
            if i == GRIPPER_INDEX:
                continue  # squeezing an object is the gripper's job; its torque limit protects it
            magnitude = abs(load[joint])
            blocked = magnitude > self.contact_load and lag[i] > self.contact_error_deg
            stalled = magnitude >= 0.95 * self.torque_limits[joint]
            self._blocked[joint] = self._blocked[joint] + 1 if blocked else 0
            self._stalled[joint] = self._stalled[joint] + 1 if stalled else 0
            if self._blocked[joint] >= self.contact_ticks:
                faults.append(
                    f"contact stop: {joint} blocked {lag[i]:.0f} deg from its goal at load {magnitude}/1000"
                )
                self._blocked[joint] = 0
            if self._stalled[joint] >= self.stall_ticks:
                faults.append(f"contact stop: {joint} pinned at its torque limit ({magnitude}/1000)")
                self._stalled[joint] = 0
        if self._tick % 30 == 0:
            self._temps = self.bus.sync_read("Present_Temperature", list(JOINTS), normalize=False)
        for joint, temp in self._temps.items():
            if temp >= self.max_temperature_c:
                faults.append(f"overheat: {joint} at {temp} C (limit {self.max_temperature_c} C)")
        return faults

    def at_rest(self) -> bool:
        joints = self.read_joints()
        return bool(np.all(np.abs(joints[:5] - REST_POSE[:5]) <= REST_TOLERANCE_DEG))

    def rest_pose(self) -> np.ndarray:
        return REST_POSE.copy()
