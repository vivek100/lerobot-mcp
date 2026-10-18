"""Cartesian planner: centimetre moves -> joint targets, with remaining-room and pose previews.

Uses the same MuJoCo arm model as the table guard. World axes: +x forward from the base, +y left, +z up.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from so101_mcp.kinematics import JOINTS, ArmKinematics
from so101_mcp.joints import GRIPPER_INDEX

# One call may not send the arm across the whole table.
MAX_TRANSLATION_M = 0.20
MAX_TILT_DEG = 45.0
MAX_ROLL_DEG = 90.0
WAYPOINT_LIFT_M = 0.06
PATH_SAMPLES = 12
ROOM_HORIZON_M = 0.22
# Folded-up working pose (~10 cm under the gripper). Same joint path `go_rest` uses, in reverse,
# so a centimetre move can leave the real-arm rest pose on the table.
UNPARK = np.array([0.0, -60.0, 60.0, 35.0, 0.0, 30.0])
# Real rest is lift ≈ −100°, elbow ≈ +92°. A pick pose with the arm stretched out is not this.
_REST_LIFT_MAX_DEG = -80.0
_REST_ELBOW_MIN_DEG = 60.0
# Task props in the MuJoCo scene. They are not on the real table, so previews leave them out.
_SCENE_PROPS = ("box", "red_cube", "green_cube", "blue_cube", "yellow_cube")
_HIDDEN_GEOM_GROUP = 5


def _folded_on_table(pose: np.ndarray) -> bool:
    return float(pose[1]) < _REST_LIFT_MAX_DEG and float(pose[2]) > _REST_ELBOW_MIN_DEG


def _rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle_rad) * k + (1 - np.cos(angle_rad)) * (k @ k)


@dataclass
class PlannedMove:
    ok: bool
    message: str
    poses: list[np.ndarray]  # joint-space waypoints including the destination; empty if refused


class MotionPlanner:
    def __init__(self, kin: ArmKinematics | None = None):
        self.kin = kin or ArmKinematics()
        self._renderer: mujoco.Renderer | None = None

    def gripper_xyz(self, pose: np.ndarray) -> np.ndarray:
        return self.kin.gripper_position(pose)

    def tilt_down_deg(self, pose: np.ndarray) -> float:
        """Fingers level = 0, straight down = 90, straight up = -90."""
        finger = self._finger(pose)
        horizontal = float(np.hypot(finger[0], finger[1]))
        return float(np.degrees(np.arctan2(-finger[2], max(horizontal, 1e-9))))

    def describe_cm(self, pose: np.ndarray) -> tuple[float, float, float, float, float]:
        xyz = self.gripper_xyz(pose)
        return (
            round(float(xyz[0] * 100), 1),
            round(float(xyz[1] * 100), 1),
            round(float((xyz[2] - self.kin.table_top_z) * 100), 1),
            round(self.tilt_down_deg(pose), 1),
            round(float(pose[4]), 1),
        )

    def solve_offset(
        self,
        seed: np.ndarray,
        forward_m: float,
        left_m: float,
        up_m: float,
        tilt_down_delta_deg: float,
        roll_delta_deg: float,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray | None:
        current = self.gripper_xyz(seed)
        target = current + np.array([forward_m, left_m, up_m], dtype=np.float64)
        finger, jaw = self._finger(seed), self._jaw(seed)
        # Yaw the hand with the gripper's heading so a side-step is a base turn, not a world-fixed wrist.
        a0 = float(np.arctan2(current[1], current[0]))
        a1 = float(np.arctan2(target[1], target[0]))
        yaw = a1 - a0
        if abs(yaw) > 1e-6:
            rot_z = _rodrigues(np.array([0.0, 0.0, 1.0]), yaw)
            finger, jaw = rot_z @ finger, rot_z @ jaw
        if abs(tilt_down_delta_deg) > 1e-6:
            heading = np.array([finger[0], finger[1], 0.0])
            if np.linalg.norm(heading) < 1e-6:
                heading = np.array([1.0, 0.0, 0.0])
            heading /= np.linalg.norm(heading)
            axis = np.array([-heading[1], heading[0], 0.0])
            rot = _rodrigues(axis, np.deg2rad(tilt_down_delta_deg))
            finger, jaw = rot @ finger, rot @ jaw
        if abs(roll_delta_deg) > 1e-6:
            rot = _rodrigues(finger, np.deg2rad(roll_delta_deg))
            jaw = rot @ jaw
        jaw -= jaw.dot(finger) * finger
        if np.linalg.norm(jaw) < 1e-9:
            return None
        jaw /= np.linalg.norm(jaw)
        return self._solve(target, finger, jaw, seed, lower, upper)

    def plan(
        self,
        seed: np.ndarray,
        forward_m: float,
        left_m: float,
        up_m: float,
        tilt_down_delta_deg: float,
        roll_delta_deg: float,
        lower: np.ndarray,
        upper: np.ndarray,
        clearance_m: float,
    ) -> PlannedMove:
        if max(abs(forward_m), abs(left_m), abs(up_m)) > MAX_TRANSLATION_M + 1e-9:
            return PlannedMove(False, f"Each translation must be within ±{MAX_TRANSLATION_M * 100:.0f} cm per call", [])
        if abs(tilt_down_delta_deg) > MAX_TILT_DEG + 1e-9:
            return PlannedMove(False, f"tilt_down_deg must be within ±{MAX_TILT_DEG:.0f}° per call", [])
        if abs(roll_delta_deg) > MAX_ROLL_DEG + 1e-9:
            return PlannedMove(False, f"roll_deg must be within ±{MAX_ROLL_DEG:.0f}° per call", [])
        if max(abs(forward_m), abs(left_m), abs(up_m), abs(tilt_down_delta_deg), abs(roll_delta_deg)) < 1e-9:
            return PlannedMove(False, "Specify at least one of forward_cm, left_cm, up_cm, tilt_down_deg, roll_deg", [])

        prefix: list[np.ndarray] = []
        note = "ok"
        start_low, _ = self.kin.lowest_point(seed)
        plan_from = seed
        # UNPARK is the reverse of go_rest. Only the folded rest pose should take it — even when
        # the real-desk offset makes rest look a few centimetres high. A stretched gripper must
        # not swing through UNPARK.
        if _folded_on_table(seed):
            via = UNPARK.copy()
            via[0] = seed[0]
            via[GRIPPER_INDEX] = seed[GRIPPER_INDEX]
            via[:5] = np.clip(via[:5], lower[:5], upper[:5])
            via_low, _ = self.kin.lowest_point(via)
            if via_low < clearance_m:
                return PlannedMove(False, "Refused: cannot lift off the table from this rest pose", [])
            prefix = [via]
            plan_from = via
            note = "left the rest pose first, then moved"

        goal = self.solve_offset(plan_from, forward_m, left_m, up_m, tilt_down_delta_deg, roll_delta_deg, lower, upper)
        if goal is None:
            return PlannedMove(False, "That pose is out of reach (or outside joint limits) from here", [])
        low, where = self.kin.lowest_point(goal)
        if low < clearance_m:
            return PlannedMove(
                False,
                f"Refused: this would bring the {where} to {low * 100:.1f} cm from the table "
                f"(minimum {clearance_m * 100:.1f} cm)",
                [],
            )
        if self._path_clear(plan_from, goal, clearance_m):
            return PlannedMove(True, note, prefix + [goal])

        hover = self.solve_offset(plan_from, 0.0, 0.0, WAYPOINT_LIFT_M, 0.0, 0.0, lower, upper)
        # The hover is an intermediate waypoint. The destination must remain the
        # original goal, not the same offset applied again from a raised pose.
        via_target = goal
        if hover is not None and via_target is not None:
            via_low, _ = self.kin.lowest_point(via_target)
            if via_low >= clearance_m and self._path_clear(plan_from, hover, clearance_m) and self._path_clear(hover, via_target, clearance_m):
                lift_note = "lifted first so the path stays above the table"
                if prefix:
                    lift_note = "left the rest pose first, then moved"
                return PlannedMove(True, lift_note, prefix + [hover, via_target])
        return PlannedMove(False, "Refused: the path between here and that pose would dip into the table; try a smaller move or separate translation and tilt", [])

    def room_left(self, pose: np.ndarray, lower: np.ndarray, upper: np.ndarray, clearance_m: float) -> list[tuple[str, float, str]]:
        axes = (
            ("forward", np.array([1.0, 0.0, 0.0])),
            ("back", np.array([-1.0, 0.0, 0.0])),
            ("left", np.array([0.0, 1.0, 0.0])),
            ("right", np.array([0.0, -1.0, 0.0])),
            ("up", np.array([0.0, 0.0, 1.0])),
            ("down", np.array([0.0, 0.0, -1.0])),
        )
        out: list[tuple[str, float, str]] = []
        for name, axis in axes:
            remaining, reason = self._search_axis(pose, axis, lower, upper, clearance_m)
            out.append((name, round(remaining * 100, 1), reason))
        return out

    def previews(self, pose: np.ndarray, width: int = 320) -> dict[str, np.ndarray]:
        """Arm-only 3D views of the current joints. The sim bin and cubes are not drawn."""
        self._hide_scene_props()
        self.kin.set_pose(pose)
        mujoco.mj_forward(self.kin.model, self.kin.data)
        if self._renderer is None or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self.kin.model, height=round(width * 3 / 4), width=width)
        option = mujoco.MjvOption()
        option.geomgroup[_HIDDEN_GEOM_GROUP] = 0
        lookat = np.array([0.18, 0.0, 0.12])
        views = {
            "preview_front": (180.0, -20.0, 0.85),
            "preview_side": (90.0, -20.0, 0.85),
            "preview_top": (180.0, -85.0, 0.70),
        }
        frames = {}
        for name, (azimuth, elevation, distance) in views.items():
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.kin.model, cam)
            cam.lookat[:] = lookat
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = elevation
            self._renderer.update_scene(self.kin.data, camera=cam, scene_option=option)
            self._add_orientation_overlay(self._renderer.scene)
            frames[name] = self._renderer.render()
            import cv2

            for x, text, color in ((4, "+X forward", (255, 85, 85)), (112, "+Y left", (70, 235, 90)), (212, "+Z up", (90, 160, 255))):
                cv2.putText(frames[name], text, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2)
                cv2.putText(frames[name], text, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
            for y, text in ((28, "Cyan: model wrist FOV (not calibrated on hardware)"), (42, "Cone length is illustrative, not measured depth")):
                cv2.putText(frames[name], text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (0, 0, 0), 2)
                cv2.putText(frames[name], text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (255, 255, 255), 1)
        return frames

    def _add_orientation_overlay(self, scene) -> None:
        """Base axes and the model wrist camera frustum; never claims physical calibration."""
        def line(start, end, color, label=""):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color, dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.001, np.asarray(start), np.asarray(end))
            # Fixed screen legend replaces MuJoCo's large, overlapping world labels.
            scene.ngeom += 1

        origin = np.array([0.0, 0.0, 0.01])
        for axis, color, label in (
            ([0.12, 0, 0], [1, 0.15, 0.15, 1], "+X"),
            ([0, 0.12, 0], [0.1, 0.8, 0.2, 1], "+Y"),
            ([0, 0, 0.12], [0.15, 0.4, 1, 1], "+Z"),
        ):
            line(origin, origin + axis, color, label)
        model, data = self.kin.model, self.kin.data
        camera_ids = [i for i in range(model.ncam) if "wrist" in model.camera(i).name.lower()]
        if not camera_ids:
            return
        camera_id = camera_ids[0]
        pos = data.cam_xpos[camera_id].copy()
        rotation = data.cam_xmat[camera_id].reshape(3, 3)
        length = 0.18
        half_y = length * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2)
        half_x = half_y * 4 / 3  # source model wrist stream is 640 x 480
        corners = [pos + rotation @ np.array([x*half_x, y*half_y, -length]) for x, y in ((-1,-1),(1,-1),(1,1),(-1,1))]
        color = [0.05, 0.75, 0.9, 1]
        for i, corner in enumerate(corners):
            line(pos, corner, color)
            line(corner, corners[(i+1) % 4], color)
        line(pos, pos + rotation @ np.array([0, 0, -length]), color, "wrist view (model)")

    def _hide_scene_props(self) -> None:
        if getattr(self, "_props_hidden", False):
            return
        model = self.kin.model
        for geom_id in range(model.ngeom):
            if model.body(model.geom_bodyid[geom_id]).name in _SCENE_PROPS:
                model.geom_group[geom_id] = _HIDDEN_GEOM_GROUP
        self._props_hidden = True

    def _finger(self, pose: np.ndarray) -> np.ndarray:
        self.kin.set_pose(pose)
        return self.kin.data.site_xmat[self.kin.gripper_site].reshape(3, 3)[:, 0].copy()

    def _jaw(self, pose: np.ndarray) -> np.ndarray:
        self.kin.set_pose(pose)
        return self.kin.data.site_xmat[self.kin.gripper_site].reshape(3, 3)[:, 2].copy()

    def _solve(
        self,
        target: np.ndarray,
        x_goal: np.ndarray,
        z_goal: np.ndarray,
        seed: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        iterations: int = 80,
        tolerance_m: float = 0.003,
        tolerance_rad: float = 0.05,
    ) -> np.ndarray | None:
        m, d = self.kin.model, self.kin.data
        dofs = np.array([m.joint(j).dofadr[0] for j in JOINTS[:5]])
        z_goal = z_goal - z_goal.dot(x_goal) * x_goal
        if np.linalg.norm(z_goal) < 1e-9:
            return None
        z_goal = z_goal / np.linalg.norm(z_goal)
        y_goal = np.cross(z_goal, x_goal)
        y_goal /= np.linalg.norm(y_goal)
        goal_axes = (x_goal, y_goal, z_goal)
        pose = np.asarray(seed, dtype=np.float64).copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iterations):
            self.kin.set_pose(pose)
            position = d.site_xpos[self.kin.gripper_site]
            axes = d.site_xmat[self.kin.gripper_site].reshape(3, 3).T
            e_pos = np.asarray(target, dtype=np.float64) - position
            e_rot = 0.5 * sum(np.cross(axes[i], goal_axes[i]) for i in range(3))
            if np.linalg.norm(e_pos) < tolerance_m and np.linalg.norm(e_rot) < tolerance_rad:
                pose[GRIPPER_INDEX] = seed[GRIPPER_INDEX]
                return pose
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, self.kin.gripper_site)
            jac = np.vstack([jacp[:, dofs], 0.15 * jacr[:, dofs]])
            error = np.concatenate([e_pos, 0.15 * e_rot])
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            pose[:5] = np.clip(pose[:5] + np.rad2deg(np.clip(step, -0.2, 0.2)), lower[:5], upper[:5])
        return None

    def _path_clear(self, start: np.ndarray, goal: np.ndarray, clearance_m: float) -> bool:
        start_low, _ = self.kin.lowest_point(start)
        floor = min(clearance_m, start_low)
        for s in np.linspace(0.0, 1.0, PATH_SAMPLES)[1:]:
            pose = start + s * (goal - start)
            low, _ = self.kin.lowest_point(pose)
            if low < floor - 1e-4:
                return False
        return True

    def _search_axis(
        self,
        pose: np.ndarray,
        axis: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        clearance_m: float,
    ) -> tuple[float, str]:
        lo, hi = 0.0, ROOM_HORIZON_M
        last_reason = "reach"
        # Down is cheap and exact from the mesh.
        if axis[2] < -0.5:
            low, where = self.kin.lowest_point(pose)
            remaining = max(0.0, low - clearance_m)
            return remaining, f"table ({where})"
        for _ in range(8):
            mid = (lo + hi) / 2
            solved = self.solve_offset(pose, *(mid * axis), 0.0, 0.0, lower, upper)
            if solved is None:
                hi = mid
                last_reason = "reach or joint limit"
                continue
            low, where = self.kin.lowest_point(solved)
            if low < clearance_m:
                hi = mid
                last_reason = f"table ({where})"
                continue
            lo = mid
        if lo < 0.004:
            return 0.0, last_reason
        return lo, last_reason
