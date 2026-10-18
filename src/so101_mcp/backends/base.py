"""Backend contract: the runtime talks to the sim and the real arm through this interface only.

All methods are called from the runtime's control thread. Backends that own GL contexts (MuJoCo
rendering) or serial ports must be created and used on that thread.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

BackendMode = Literal["sim", "real"]


@dataclass(frozen=True)
class JointLimits:
    """Soft limits in joint units (degrees, gripper 0-100), ordered like ``JOINTS``."""

    lower: np.ndarray
    upper: np.ndarray


@dataclass
class TaskSignals:
    """Ground-truth task signals; only simulation can provide them."""

    available: bool
    reward: float | None = None
    success: bool | None = None
    details: dict[str, float] = field(default_factory=dict)


class RobotBackend(ABC):
    mode: BackendMode
    name: str
    camera_names: tuple[str, ...]
    fps: int

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def joint_limits(self) -> JointLimits: ...

    @abstractmethod
    def read_joints(self) -> np.ndarray:
        """Measured joint positions, ordered like ``JOINTS``."""

    @abstractmethod
    def command(self, target: np.ndarray) -> None:
        """Apply absolute joint targets for one control period (sim advances physics here)."""

    @abstractmethod
    def render(self, camera: str) -> np.ndarray:
        """One RGB uint8 HxWx3 frame from a named camera."""

    @abstractmethod
    def reset(self, seed: int | None) -> None:
        """Sim: restore and randomize the scene. Real: not supported (raise)."""

    def task_signals(self) -> TaskSignals:
        return TaskSignals(available=False)

    def camera_geometry(self, camera: str) -> dict | None:
        """Calibrated pinhole camera pose in model world. Real cameras require calibration first."""
        return None

    def contact_faults(self) -> list[str]:
        """Called every tick; any message stops the active job and holds the measured pose."""
        return []

    def rest_pose(self) -> np.ndarray:
        """Safe parking pose in joint units (also where the real arm may release torque)."""
        return np.array([0.0, -60.0, 60.0, 35.0, 0.0, 30.0])

    def at_rest(self) -> bool:
        return True

    def sim_time(self) -> float | None:
        return None

    def object_positions(self) -> dict[str, np.ndarray]:
        """Sim: ground-truth xyz (m) of movable objects. Real: none."""
        return {}

    def place_object(self, name: str, x: float, y: float, z: float | None = None) -> None:
        raise RuntimeError("place_object is only available in simulation")

    def attach_viewer(self) -> None:
        """Optional 3D viewer driven from the control thread."""

    def sync_viewer(self) -> None:
        """Called once per control tick after ``command``."""
