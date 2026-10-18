"""Joint naming and units shared by every backend.

Units follow LeRobot's ``so101_follower`` with ``use_degrees=True``: degrees for the five arm joints,
0-100 for the gripper opening. Keys carry the ``.pos`` suffix used by LeRobot observations/actions.
"""

from __future__ import annotations

import numpy as np

JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_KEYS: tuple[str, ...] = tuple(f"{j}.pos" for j in JOINTS)
GRIPPER_INDEX = 5


def to_dict(values: np.ndarray) -> dict[str, float]:
    return {key: float(v) for key, v in zip(JOINT_KEYS, values)}


def from_dict(values: dict[str, float]) -> np.ndarray:
    return np.array([values[key] for key in JOINT_KEYS], dtype=np.float64)
