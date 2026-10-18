"""MuJoCo backend: LeRobot's SO-101 tabletop EnvHub environment, driven like a real arm.

The env's own ``step`` renders every camera on every call (~21 ms for two cameras here). The runtime
needs physics at 30 Hz but camera frames only on demand, so this backend steps the env's MuJoCo world
directly and renders cameras separately. It relies on the env's ``_World`` helpers (``to_ctrl``,
``joint_state``, ``render``, ``cubes_in_box``), so the env revision is pinned.
"""

from __future__ import annotations

import contextlib

import mujoco
import numpy as np

from so101_mcp.backends.base import JointLimits, RobotBackend, TaskSignals
from so101_mcp.joints import GRIPPER_INDEX

HUB_REPO = "lerobot/so101-tabletop-mujoco"
HUB_REVISION = "12e504e860c369aad1a9a9eab3fdfbac658ed6dd"
CUBES = 4
OBJECTS = ("red_cube", "green_cube", "blue_cube", "yellow_cube")
CUBE_HALF = 0.015


class SimBackend(RobotBackend):
    mode = "sim"
    name = "so101-tabletop-mujoco"

    def __init__(
        self,
        seed: int | None = 0,
        camera_shadows: bool = False,
        limit_margin_deg: float = 3.0,
        revision: str = HUB_REVISION,
    ):
        self.seed = seed
        self.camera_shadows = camera_shadows
        self.limit_margin_deg = limit_margin_deg
        self.revision = revision
        self._env = None
        self._world = None
        self._viewer = None
        self.fps = 30
        self.camera_names = ("front", "wrist")

    def connect(self) -> None:
        from lerobot.envs.factory import make_env

        envs = make_env(f"{HUB_REPO}@{self.revision}:env.py", n_envs=1, trust_remote_code=True)
        self._vector_env = next(iter(envs.values()))[0]
        self._env = self._vector_env.envs[0].unwrapped
        self._env.reset(seed=self.seed)
        self._world = self._env._ensure_world()
        self.fps = int(self._env.fps)
        self._camera_map = dict(self._env.cameras)
        self.camera_names = tuple(self._camera_map)
        flags = self._world.renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(self.camera_shadows)
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(self.camera_shadows)

    def disconnect(self) -> None:
        if self._viewer is not None:
            with contextlib.suppress(Exception):
                self._viewer.close()
            self._viewer = None
        if self._env is not None:
            self._env.close()
            self._env = None
            self._world = None

    def joint_limits(self) -> JointLimits:
        w = self._world
        lower = np.rad2deg(w.lo) + self.limit_margin_deg
        upper = np.rad2deg(w.hi) - self.limit_margin_deg
        lower[GRIPPER_INDEX], upper[GRIPPER_INDEX] = 0.0, 100.0
        return JointLimits(lower.astype(np.float64), upper.astype(np.float64))

    def read_joints(self) -> np.ndarray:
        return self._world.joint_state().astype(np.float64)

    def command(self, target: np.ndarray) -> None:
        w = self._world
        with self._viewer_lock():
            w.data.ctrl[w.actuators] = w.to_ctrl(np.asarray(target, dtype=np.float64))
            mujoco.mj_step(w.model, w.data, nstep=w.substeps)

    def render(self, camera: str) -> np.ndarray:
        if camera not in self._camera_map:
            raise ValueError(f"Unknown camera {camera!r}; available: {list(self._camera_map)}")
        return self._world.render(self._camera_map[camera])

    def reset(self, seed: int | None) -> None:
        with self._viewer_lock():
            self._env.reset(seed=seed)

    def camera_geometry(self, camera: str) -> dict:
        w = self._world
        camera_id = w.model.camera(self._camera_map[camera]).id
        return {
            "source": "simulator_calibration", "position_m": w.data.cam_xpos[camera_id].tolist(),
            "rotation_camera_to_world": w.data.cam_xmat[camera_id].reshape(3, 3).tolist(),
            "vertical_fov_deg": float(w.model.cam_fovy[camera_id]),
        }

    def task_signals(self) -> TaskSignals:
        in_box = self._world.cubes_in_box()
        return TaskSignals(
            available=True,
            reward=in_box / CUBES,
            success=in_box == CUBES,
            details={"cubes_in_box": float(in_box), "cubes_total": float(CUBES)},
        )

    def sim_time(self) -> float | None:
        return float(self._world.data.time)

    def object_positions(self) -> dict[str, np.ndarray]:
        m, d = self._world.model, self._world.data
        return {name: d.xpos[m.body(name).id].copy() for name in OBJECTS}

    def place_object(self, name: str, x: float, y: float, z: float | None = None) -> None:
        """Teleport a cube (upright, at rest); z defaults to resting on the table."""
        if name not in OBJECTS:
            raise ValueError(f"Unknown object {name!r}; available: {list(OBJECTS)}")
        m, d = self._world.model, self._world.data
        joint = m.body_jntadr[m.body(name).id]
        qpos, dof = m.jnt_qposadr[joint], m.jnt_dofadr[joint]
        with self._viewer_lock():
            d.qpos[qpos:qpos + 3] = (x, y, CUBE_HALF if z is None else z)
            d.qpos[qpos + 3:qpos + 7] = (1.0, 0.0, 0.0, 0.0)
            d.qvel[dof:dof + 6] = 0.0
            mujoco.mj_forward(m, d)

    def attach_viewer(self) -> None:
        import mujoco.viewer

        model, data = self._world.model, self._world.data
        self._viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        mount = data.site_xpos[model.site("so101_mount").id]
        box = data.xpos[self._world.box_body]
        with self._viewer.lock():
            cam = self._viewer.cam
            cam.lookat[:] = (mount + box) / 2 + np.array([0.0, 0.0, 0.08])
            cam.distance = 0.85
            cam.elevation = -28.0
            cam.azimuth = 135.0

    def sync_viewer(self) -> None:
        if self._viewer is None:
            return
        if not self._viewer.is_running():
            self._viewer = None
            return
        self._viewer.sync()

    def _viewer_lock(self):
        return self._viewer.lock() if self._viewer is not None else contextlib.nullcontext()
