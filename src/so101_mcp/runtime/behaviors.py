"""Behaviours produce joint targets tick by tick; the runtime owns timing and safety.

A behaviour never talks to the backend. Each tick it receives the measured joints and the elapsed
time and returns a target, or ``None`` when it has finished.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from so101_mcp.joints import GRIPPER_INDEX, JOINT_KEYS


@dataclass
class BehaviorOutcome:
    """What a finished behaviour reports for verification."""

    reached: bool
    final_error: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None  # set when the behaviour gave up; the job fails with this message


class Behavior(ABC):
    kind: str
    # A preemptible behaviour (an ongoing expressive state) is cancelled by the next motion request
    # instead of making it fail as busy.
    preemptible: bool = False
    # Table-guard clearance for this job; None keeps the default. Grasping needs the jaws near the table.
    table_clearance_m: float | None = None
    # Streaming behaviours (policies) hold the last safe pose for a refused tick instead of failing the job.
    hold_on_table_guard: bool = False

    @abstractmethod
    def start(self, measured: np.ndarray) -> None: ...

    @abstractmethod
    def step(self, measured: np.ndarray, t: float) -> np.ndarray | None: ...

    @abstractmethod
    def progress(self, t: float) -> float: ...

    @abstractmethod
    def outcome(self, measured: np.ndarray) -> BehaviorOutcome: ...

    def finish(self) -> None:
        """Called once when the job ends for any reason (completed, failed, cancelled)."""

    def metrics(self) -> dict[str, float]:
        """Live numbers published on the job each tick (e.g. tracking error)."""
        return {}


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# Peak velocity of a min-jerk profile relative to its average velocity.
MIN_JERK_PEAK = 1.875


def min_jerk_duration(origin: np.ndarray, goal: np.ndarray, max_speed: np.ndarray) -> float:
    """Shortest min-jerk duration whose peak speed stays under every joint's cap."""
    return MIN_JERK_PEAK * float(np.max(np.abs(goal - origin) / max_speed))


class JogBehavior(Behavior):
    """Smoothly move some joints to absolute or relative targets, then wait to settle.

    ``targets`` maps joint index -> value. Unlisted joints hold their starting position.
    Duration comes from the largest move divided by ``speed`` (average deg/s; gripper units/s),
    stretched so the profile's peak stays under ``max_speed``. Otherwise the safety limiter would clip
    each joint separately and bend the straight joint-space path (on the real arm, toward the table).
    """

    kind = "jog"

    def __init__(
        self,
        targets: dict[int, float],
        relative: bool,
        speed: float,
        limits_lower: np.ndarray,
        limits_upper: np.ndarray,
        tolerance: float = 2.0,
        settle_timeout: float = 1.5,
        max_speed: np.ndarray | None = None,
    ):
        self.targets = targets
        self.relative = relative
        self.speed = speed
        self.max_speed = max_speed
        self.lower, self.upper = limits_lower, limits_upper
        self.tolerance = tolerance
        self.settle_timeout = settle_timeout
        self.notes: list[str] = []

    def start(self, measured: np.ndarray) -> None:
        self.origin = measured.copy()
        goal = measured.copy()
        for index, value in self.targets.items():
            goal[index] = (measured[index] + value) if self.relative else value
        clipped = np.clip(goal, self.lower, self.upper)
        if np.any(clipped != goal):
            self.notes.append("goal_clamped_to_joint_limits")
        self.goal = clipped
        moves = np.abs(self.goal - self.origin)
        moves[GRIPPER_INDEX] *= 0.5  # gripper units move faster than degrees
        largest = float(moves.max())
        self.duration = max(largest / self.speed, 0.2)
        if self.max_speed is not None:
            self.duration = max(self.duration, min_jerk_duration(self.origin, self.goal, self.max_speed))

    def step(self, measured: np.ndarray, t: float) -> np.ndarray | None:
        if t <= self.duration:
            return self.origin + min_jerk(t / self.duration) * (self.goal - self.origin)
        settled = np.all(np.abs(self.goal - measured) <= self.tolerance)
        if settled or t > self.duration + self.settle_timeout:
            return None
        return self.goal

    def progress(self, t: float) -> float:
        return min(t / self.duration, 1.0)

    def outcome(self, measured: np.ndarray) -> BehaviorOutcome:
        error = self.goal - measured
        moved = {i for i in range(len(error)) if abs(self.goal[i] - self.origin[i]) > 1e-6} | set(self.targets)
        reached = all(abs(error[i]) <= self.tolerance for i in moved)
        notes = list(self.notes)
        if not reached:
            notes.append("target_not_reached_within_tolerance (blocked, contact, or limit)")
        return BehaviorOutcome(
            reached=reached,
            final_error={JOINT_KEYS[i]: round(float(error[i]), 2) for i in sorted(moved)},
            notes=notes,
        )


class PathBehavior(Behavior):
    """Follow joint-space waypoints (from the Cartesian planner) on one min-jerk profile, then settle.

    The path is parameterized by cumulative joint travel; the duration is the shortest that keeps every
    joint's peak speed under ``max_speed`` on every segment. Fixed speed: callers do not choose it.
    """

    kind = "move"

    def __init__(self, waypoints: list[np.ndarray], max_speed: np.ndarray, tolerance: float = 2.0, settle_timeout: float = 1.5):
        self.waypoints = [np.asarray(w, dtype=np.float64) for w in waypoints]
        self.max_speed = np.asarray(max_speed, dtype=np.float64)
        self.tolerance = tolerance
        self.settle_timeout = settle_timeout

    def start(self, measured: np.ndarray) -> None:
        points = [measured.copy(), *self.waypoints]
        for p in points[1:]:
            p[GRIPPER_INDEX] = measured[GRIPPER_INDEX]
        self.points = np.array(points)
        self.goal = self.points[-1].copy()
        lengths = np.linalg.norm(np.diff(self.points[:, :GRIPPER_INDEX], axis=0), axis=1)
        total = float(lengths.sum())
        if total < 1e-6:
            self.knots = np.linspace(0.0, 1.0, len(self.points))
            self.duration = 0.2
            return
        self.knots = np.concatenate([[0.0], np.cumsum(lengths) / total])
        rates = np.abs(np.diff(self.points, axis=0)) / np.maximum(np.diff(self.knots), 1e-9)[:, None]
        self.duration = max(MIN_JERK_PEAK * float(np.max(rates / self.max_speed)), 0.4)

    def _at(self, s: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.knots, s, side="right") - 1, 0, len(self.points) - 2))
        span = self.knots[i + 1] - self.knots[i]
        u = 0.0 if span <= 0 else (s - self.knots[i]) / span
        return self.points[i] + u * (self.points[i + 1] - self.points[i])

    def step(self, measured: np.ndarray, t: float) -> np.ndarray | None:
        if t <= self.duration:
            return self._at(min_jerk(t / self.duration))
        settled = np.all(np.abs(self.goal - measured)[:GRIPPER_INDEX] <= self.tolerance)
        if settled or t > self.duration + self.settle_timeout:
            return None
        return self.goal

    def progress(self, t: float) -> float:
        return min(t / self.duration, 1.0)

    def outcome(self, measured: np.ndarray) -> BehaviorOutcome:
        error = self.goal - measured
        reached = bool(np.all(np.abs(error[:GRIPPER_INDEX]) <= self.tolerance))
        notes = [] if reached else ["target_not_reached_within_tolerance (blocked, contact, or limit)"]
        return BehaviorOutcome(
            reached=reached,
            final_error={JOINT_KEYS[i]: round(float(error[i]), 2) for i in range(GRIPPER_INDEX)},
            notes=notes,
        )


class SequenceBehavior(Behavior):
    """Runs child behaviours back to back; each starts from the previous one's commanded goal."""

    kind = "sequence"

    def __init__(self, children: list[Behavior]):
        self.children = children

    def start(self, measured: np.ndarray) -> None:
        self.index = 0
        self.t0 = 0.0
        self.children[0].start(measured)

    def step(self, measured: np.ndarray, t: float) -> np.ndarray | None:
        while True:
            child = self.children[self.index]
            target = child.step(measured, t - self.t0)
            if target is not None:
                self.last = target
                return target
            if self.index == len(self.children) - 1:
                return None
            self.index += 1
            self.t0 = t
            self.children[self.index].start(getattr(child, "goal", measured).copy())

    def progress(self, t: float) -> float:
        current = self.children[self.index].progress(t - self.t0)
        return (self.index + current) / len(self.children)

    def outcome(self, measured: np.ndarray) -> BehaviorOutcome:
        return self.children[-1].outcome(measured)

    def finish(self) -> None:
        for child in self.children:
            child.finish()
