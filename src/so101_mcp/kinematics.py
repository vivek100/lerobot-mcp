"""Forward kinematics of the SO-101 from the same MuJoCo model the simulation uses.

The tabletop scene and the Menagerie arm are compiled without a renderer or physics stepping, so
this works in real mode too. Poses are in runtime units: degrees for joints, 0-100 for the gripper.
The real arm uses the same joint convention (verified by rendering the real rest pose in sim).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

HUB_REPO = "lerobot/so101-tabletop-mujoco"
HUB_REVISION = "12e504e860c369aad1a9a9eab3fdfbac658ed6dd"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# Every moving link that can reach the table (the base and shoulder are bolted to it).
GUARDED_BODIES = ("upper_arm", "lower_arm", "wrist", "gripper", "camera_mount", "moving_jaw_so101_v1")
# Sim tabletop is MuJoCo z = 0. On the real desk a ruler read 2.5 cm of air under the
# gripper at the stretched pick pose while the mesh reported -0.7 cm (through the
# modelled table). Real-arm kinematics use REAL_TABLE_TOP_Z so clearance matches the ruler.
TABLE_TOP_Z = 0.0
REAL_TABLE_TOP_Z = -0.032


@lru_cache(maxsize=1)
def scene_root() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HUB_REPO, revision=HUB_REVISION))


def build_model() -> mujoco.MjModel:
    root = scene_root()
    spec = mujoco.MjSpec.from_file(str(root / "so101_tabletop.xml"))
    arm = mujoco.MjSpec.from_file(str(root / "robotstudio_so101" / "so101.xml"))
    spec.attach(arm, site="so101_mount", prefix="")
    return spec.compile()


class ArmKinematics:
    def __init__(self, table_top_z: float = TABLE_TOP_Z):
        self.table_top_z = table_top_z
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = np.array([m.joint(j).qposadr[0] for j in JOINTS])
        self.gripper_lo, self.gripper_hi = m.jnt_range[m.joint("gripper").id]
        bodies = {m.body(name).id: name for name in GUARDED_BODIES}
        self._geoms: list[tuple[int, np.ndarray, str]] = []
        for g in range(m.ngeom):
            body = int(m.geom_bodyid[g])
            if body not in bodies:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                points = m.mesh_vert[start : start + count].astype(np.float64)
            else:
                center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
                points = center + signs * half
            self._geoms.append((g, points, bodies[body]))
        self.gripper_site = m.site("gripperframe").id

    def set_pose(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=np.float64)
        q = np.deg2rad(pose)
        q[5] = self.gripper_lo + pose[5] / 100.0 * (self.gripper_hi - self.gripper_lo)
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)

    def lowest_point(self, pose: np.ndarray) -> tuple[float, str]:
        """Height (m) above the table top of the lowest point on any moving link, and that link."""
        self.set_pose(pose)
        best, where = np.inf, ""
        for g, points, body in self._geoms:
            z = self.data.geom_xpos[g, 2] + points @ self.data.geom_xmat[g].reshape(3, 3)[2]
            low = float(z.min())
            if low < best:
                best, where = low, body
        return best - self.table_top_z, where

    def gripper_position(self, pose: np.ndarray) -> np.ndarray:
        """World position (m) of the gripper frame site (between the jaws)."""
        self.set_pose(pose)
        return self.data.site_xpos[self.gripper_site].copy()

    def solve_grasp(
        self,
        target: np.ndarray,
        jaw_yaw: float,
        seed: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        tilts_deg: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0),
    ) -> np.ndarray | None:
        """Most vertical reachable grasp pose: the first tilt in ``tilts_deg`` that solves."""
        for tilt in tilts_deg:
            pose = self.solve_top_down(target, jaw_yaw, seed, lower, upper, tilt_deg=tilt)
            if pose is not None:
                return pose
        return None

    def solve_top_down(
        self,
        target: np.ndarray,
        jaw_yaw: float,
        seed: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        tilt_deg: float = 0.0,
        iterations: int = 200,
        tolerance_m: float = 0.002,
        tolerance_rad: float = 0.03,
    ) -> np.ndarray | None:
        """Arm pose putting the gripper site at ``target`` with the fingers pointing down.

        The gripper site's x axis runs along the fingers and its z axis is the jaw opening direction.
        ``tilt_deg`` leans the fingers away from the base (in the vertical plane through the base and
        the target); ``jaw_yaw`` (rad, world frame) sets which horizontal direction the jaws open
        along, projected perpendicular to the fingers. Joint limits (degrees) are respected; the
        gripper value is copied from ``seed``. Returns None if the damped least-squares solve does
        not converge.
        """
        m, d = self.model, self.data
        dofs = np.array([m.joint(j).dofadr[0] for j in JOINTS[:5]])
        radial = np.array([target[0], target[1], 0.0])
        radial /= max(np.linalg.norm(radial), 1e-9)
        tilt = np.deg2rad(tilt_deg)
        x_goal = np.sin(tilt) * radial + np.array([0.0, 0.0, -np.cos(tilt)])
        z_goal = np.array([np.cos(jaw_yaw), np.sin(jaw_yaw), 0.0])
        z_goal -= z_goal.dot(x_goal) * x_goal
        z_goal /= np.linalg.norm(z_goal)
        goal_axes = (x_goal, np.cross(z_goal, x_goal), z_goal)
        pose = np.asarray(seed, dtype=np.float64).copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iterations):
            self.set_pose(pose)
            position = d.site_xpos[self.gripper_site]
            axes = d.site_xmat[self.gripper_site].reshape(3, 3).T
            e_pos = np.asarray(target, dtype=np.float64) - position
            e_rot = 0.5 * sum(np.cross(axes[i], goal_axes[i]) for i in range(3))
            if np.linalg.norm(e_pos) < tolerance_m and np.linalg.norm(e_rot) < tolerance_rad:
                return pose
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, jacp, jacr, self.gripper_site)
            jac = np.vstack([jacp[:, dofs], 0.1 * jacr[:, dofs]])
            error = np.concatenate([e_pos, 0.1 * e_rot])
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            pose[:5] = np.clip(pose[:5] + np.rad2deg(np.clip(step, -0.2, 0.2)), lower[:5], upper[:5])
        return None
