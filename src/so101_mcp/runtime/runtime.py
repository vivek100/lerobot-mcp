"""RobotRuntime: the single owner of one arm.

One control thread creates the backend, then runs a fixed-rate loop: drain commands, read joints,
ask the active behaviour for a target, pass it through the SafetyLimiter, command the backend,
serve render requests, publish a status snapshot. Other threads (the MCP server) never touch the
backend; they post commands and wait on futures. At most one job owns motion at a time.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

from so101_mcp.backends.base import JointLimits, RobotBackend, TaskSignals
from so101_mcp.cameras import LAPTOP_WEBCAM, FrameGrabber
from so101_mcp.kinematics import REAL_TABLE_TOP_Z, TABLE_TOP_Z, ArmKinematics
from so101_mcp.planner import UNPARK, MotionPlanner
from so101_mcp.policies import REGISTRY as POLICIES
from so101_mcp.policies import LoadedPolicy
from so101_mcp.runtime.animation import LIBRARY, TrajectoryBehavior, fit_motion
from so101_mcp.runtime.policy_behavior import PolicyBehavior
from so101_mcp.runtime.behaviors import Behavior, JogBehavior, PathBehavior, SequenceBehavior
from so101_mcp.runtime.jobs import Job, JobStore
from so101_mcp.runtime.safety import (
    TABLE_CLEARANCE_M,
    SafetyLimiter,
    SpeedLimits,
    TableGuard,
    TableGuardError,
    UnsafeTargetError,
)
from so101_mcp.runtime.talk import TalkScript, talk_base
from so101_mcp.runtime.tracking import ColorDetector, FaceDetector, TrackingBehavior, yunet_path

# On the real arm the "front" camera is the laptop webcam.
LAPTOP_WEBCAM_ALIAS = "front"

log = logging.getLogger(__name__)
RAISED_WAYPOINT = UNPARK


@dataclass
class Snapshot:
    mode: str
    joints: np.ndarray
    frames: dict[str, np.ndarray]
    previews: dict[str, np.ndarray]
    faults: list[str]
    lowest_m: float
    room: list[tuple[str, float, str]]
    forward_cm: float
    left_cm: float
    up_cm: float
    tilt_down_deg: float
    roll_deg: float
    camera_geometry: dict[str, dict] = field(default_factory=dict)


class MotionBusyError(RuntimeError):
    pass


@dataclass
class RuntimeStatus:
    mode: str
    backend: str
    connected: bool
    fps: int
    measured_hz: float
    overruns: int
    tick: int
    sim_time: float | None
    joints: np.ndarray
    commanded: np.ndarray
    limits: JointLimits
    cameras: tuple[str, ...]
    active_job_id: str | None
    task: TaskSignals
    faults: list[str] = field(default_factory=list)
    objects: dict[str, np.ndarray] = field(default_factory=dict)  # sim ground truth, xyz in metres


class RobotRuntime:
    def __init__(
        self,
        backend_factory: Callable[[], RobotBackend],
        viewer: bool = False,
        speed: SpeedLimits = SpeedLimits(),
        max_job_seconds: float = 120.0,
        realtime: bool = True,
        table_guard: bool = True,
        reach_tolerance_deg: float = 2.0,
    ):
        self.reach_tolerance = reach_tolerance_deg
        self.planner = MotionPlanner()
        self.policies: dict[str, LoadedPolicy] = {}
        self.policy_threads = 4
        self._policy_lock = threading.Lock()
        self._factory = backend_factory
        self._viewer = viewer
        self._speed = speed
        self._use_table_guard = table_guard
        self.max_job_seconds = max_job_seconds
        self.realtime = realtime
        self.jobs = JobStore()
        self._commands: queue.Queue[tuple[str, tuple, Future]] = queue.Queue()
        self._shutdown = threading.Event()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._status: RuntimeStatus | None = None
        self._status_lock = threading.Lock()
        self._faults: deque[str] = deque(maxlen=20)
        self._thread = threading.Thread(target=self._run, name="so101-control", daemon=True)

    # ---------- lifecycle ----------
    def start(self, timeout: float = 120.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("Backend did not connect in time")
        if self._startup_error is not None:
            raise RuntimeError("Backend failed to start") from self._startup_error

    def shutdown(self, timeout: float = 5.0, park_seconds: float = 30.0) -> None:
        """Real arm: park at the rest pose first, so the backend may release torque without a drop."""
        try:
            if self._thread.is_alive() and self.status().mode == "real" and not self._call("at_rest", ()):
                self._call("stop", ())
                job = self.start_rest()
                deadline = time.monotonic() + park_seconds
                while not job.done and time.monotonic() < deadline:
                    time.sleep(0.05)
                print(
                    f"[so101-runtime] park job {job.state}, reached={job.reached}, error={job.error}, "
                    f"at_rest={self._call('at_rest', ())}",
                    flush=True,
                )
        except Exception:  # noqa: BLE001
            log.exception("Parking before shutdown failed")
        self._shutdown.set()
        self._thread.join(timeout)

    # ---------- thread-safe API ----------
    def camera_frame(self, camera: str) -> np.ndarray:
        """One image, captured on the backend thread without planner work."""
        return self._call("camera_frame", (camera,), timeout=5.0)

    def _cmd_camera_frame(self, camera: str) -> np.ndarray:
        return self.backend.render(camera)

    def status(self) -> RuntimeStatus:
        with self._status_lock:
            if self._status is None:
                raise RuntimeError("Runtime not started")
            return self._status

    def start_jog(self, targets: dict[int, float], relative: bool, speed: float) -> Job:
        speed = float(min(max(speed, 1.0), self._speed.arm_deg_per_s))
        params = {"targets": {str(k): v for k, v in targets.items()}, "relative": relative, "speed": speed}

        def make(limits: JointLimits) -> Behavior:
            return JogBehavior(
                targets, relative, speed, limits.lower, limits.upper,
                tolerance=self.reach_tolerance, max_speed=self._speed.velocity(),
            )

        return self._call("start_job", ("jog", params, make))

    def start_cartesian(
        self,
        forward_m: float,
        left_m: float,
        up_m: float,
        tilt_down_deg: float,
        roll_deg: float,
    ) -> tuple[Job | None, str, bool]:
        """Plan a centimetre-space move on this thread, then start it. Returns (job, message, ok)."""
        seed = self.status().commanded.copy()
        clearance = self.safety.table_guard.clearance if self.safety.table_guard is not None else TABLE_CLEARANCE_M
        planned = self.planner.plan(
            seed, forward_m, left_m, up_m, tilt_down_deg, roll_deg,
            self.limits.lower, self.limits.upper, clearance,
        )
        if not planned.ok:
            return None, planned.message, False
        params = {
            "forward_cm": round(forward_m * 100, 1), "left_cm": round(left_m * 100, 1),
            "up_cm": round(up_m * 100, 1), "tilt_down_deg": round(tilt_down_deg, 1),
            "roll_deg": round(roll_deg, 1), "note": planned.message,
        }
        return self.start_path(planned.poses, params), planned.message, True

    def start_gripper(self, openness: float) -> Job:
        return self.start_jog({5: float(openness)}, relative=False, speed=self._speed.gripper_units_per_s)

    def wait_job(self, job: Job, timeout: float = 90.0) -> Job:
        deadline = time.monotonic() + timeout
        while not job.done and time.monotonic() < deadline:
            time.sleep(0.05)
        return job

    def snapshot(self, previews: bool = True) -> Snapshot:
        joints, frames, faults, mode, camera_geometry = self._call("snapshot_raw", (), timeout=20.0)
        clearance = self.safety.table_guard.clearance if self.safety.table_guard is not None else TABLE_CLEARANCE_M
        lowest, _ = self.planner.kin.lowest_point(joints)
        room = self.planner.room_left(joints, self.limits.lower, self.limits.upper, clearance)
        fwd, left, up, tilt, roll = self.planner.describe_cm(joints)
        return Snapshot(
            mode=mode, joints=joints, frames=frames,
            previews=self.planner.previews(joints) if previews else {},
            faults=faults, lowest_m=lowest, room=room,
            forward_cm=fwd, left_cm=left, up_cm=up, tilt_down_deg=tilt, roll_deg=roll,
            camera_geometry=camera_geometry,
        )

    def start_path(self, waypoints: list[np.ndarray], params: dict) -> Job:
        """Follow planner waypoints at the fixed move speed (60% of each joint's cap)."""

        def make(limits: JointLimits) -> Behavior:
            return PathBehavior(waypoints, 0.6 * self._speed.velocity(), tolerance=max(3.0, self.reach_tolerance))

        return self._call("start_job", ("move", params, make))

    def start_rest(self, speed: float | None = None) -> Job:
        """Lift to a raised waypoint, then fold to the backend's rest pose.

        A straight joint-space line from a low reaching pose to the rest pose skims the table, so the
        table guard would refuse it; the waypoint keeps the whole path clear.
        """
        speed = float(min(speed or self._speed.arm_deg_per_s, self._speed.arm_deg_per_s))

        def make(limits: JointLimits) -> Behavior:
            rest = self.backend.rest_pose()
            tolerance = max(3.0, self.reach_tolerance)

            def jog(pose: np.ndarray) -> JogBehavior:
                return JogBehavior(
                    dict(enumerate(pose.tolist())), False, speed, limits.lower, limits.upper,
                    tolerance=tolerance, max_speed=self._speed.velocity(),
                )

            waypoint = RAISED_WAYPOINT.copy()
            waypoint[0] = rest[0]
            if np.allclose(waypoint, rest):
                return jog(rest)
            return SequenceBehavior([jog(waypoint), jog(rest)])

        return self._call("start_job", ("rest", {"speed": speed}, make))

    def start_animation(self, name: str, repeat: int = 1, seconds: float | None = None) -> Job:
        """Play a library animation: glide to its base pose, play it fitted to the speed budget, settle.

        One-shot gestures play ``repeat`` times; ongoing states (``loopable``) last ``seconds``.
        """
        if name not in LIBRARY:
            raise ValueError(f"Unknown animation {name!r}; available: {sorted(LIBRARY)}")
        anim = LIBRARY[name]
        total = float(seconds) if anim.loopable and seconds else anim.cycle_s * max(1, repeat)
        if total > self.max_job_seconds - 15:
            raise ValueError(f"Animation would last {total:.0f}s; the limit is {self.max_job_seconds - 15:.0f}s")

        def make(limits: JointLimits) -> Behavior:
            motion = fit_motion(
                anim.base, anim.make(total), total, self.backend.fps, limits.lower, limits.upper,
                self._speed.velocity(), self._speed.acceleration(),
            )
            return TrajectoryBehavior(
                "animation", motion, self._speed.velocity(), tolerance=max(3.0, self.reach_tolerance),
                preemptible=anim.loopable,
            )

        kind = "express" if anim.loopable else "animation"
        return self._call("start_job", (kind, {"name": name, "seconds": round(total, 2)}, make))

    def start_talk(self, script: TalkScript) -> Job:
        """Play a prepared utterance; speech audio (if any) starts on the tick the mouth motion starts."""

        def make(limits: JointLimits) -> Behavior:
            # Never slowed down: the motion must stay in sync with the audio.
            motion = fit_motion(
                talk_base(), script.offsets(self._speed.gripper_units_per_s), script.duration,
                self.backend.fps, limits.lower, limits.upper, self._speed.velocity(),
                self._speed.acceleration(), allow_slowdown=False,
            )
            return TrajectoryBehavior(
                "talk", motion, self._speed.velocity(), tolerance=max(3.0, self.reach_tolerance),
                on_play=script.play, on_finish=script.finish,
            )

        params = {"text": script.text[:200], "mood": script.mood, "seconds": round(script.duration, 2),
                  "voice": script.voice_name}
        return self._call("start_job", ("talk", params, make))

    def start_track(
        self, target: str, color: str | None, camera: str, seconds: float, invert: bool = False
    ) -> Job:
        """Follow a face or a coloured object with a camera (see runtime/tracking.py for the two modes).

        ``camera="webcam"`` in sim opens the laptop webcam directly ("hybrid": real detector, virtual arm).
        """
        status = self.status()
        if camera == "webcam" and LAPTOP_WEBCAM_ALIAS in status.cameras and status.mode == "real":
            camera = LAPTOP_WEBCAM_ALIAS
        grabber = None
        if camera in status.cameras:
            source = lambda: self.backend.render(camera)  # noqa: E731 - runs on the control thread
        elif camera == "webcam":
            grabber = FrameGrabber(LAPTOP_WEBCAM)
            source = grabber.latest
        else:
            raise ValueError(f"Unknown camera {camera!r}; available: {list(status.cameras)} or 'webcam'")
        try:
            if target == "face":
                detector = FaceDetector(yunet_path())
            elif target == "color":
                detector = ColorDetector(color or "red")
            else:
                raise ValueError(f"Unknown target {target!r}; use 'face' or 'color'")
            # Positive pan swings the arm toward its right (-y). A camera facing the arm (front) sees targets on
            # the arm's left at image right; the laptop webcam in sim is treated as standing beside the arm,
            # looking the same way, so image right is the arm's right.
            sign = (1.0 if camera == "webcam" else -1.0) * (-1.0 if invert else 1.0)

            def make(limits: JointLimits) -> Behavior:
                return TrackingBehavior(
                    source, detector, "servo" if camera == "wrist" else "point", seconds,
                    limits.lower, limits.upper, self.backend.fps, point_sign=sign,
                    on_finish=grabber.close if grabber else None,
                )

            params = {"target": target, "color": color, "camera": camera, "seconds": seconds}
            return self._call("start_job", ("track", params, make))
        except BaseException:
            if grabber is not None:
                grabber.close()
            raise

    def load_policy(self, name: str) -> LoadedPolicy:
        """Download (first time) and load a registered policy on CPU; cached, so later calls are instant."""
        if name not in POLICIES:
            raise ValueError(f"Unknown policy {name!r}; available: {sorted(POLICIES)}")
        with self._policy_lock:
            if name not in self.policies:
                import torch

                # Leave cores for the control loop, physics and rendering (inference is ~285 ms on 4 threads).
                torch.set_num_threads(self.policy_threads)
                self.policies[name] = LoadedPolicy.load(POLICIES[name])
            return self.policies[name]

    def start_policy(self, name: str, seconds: float, stop_on_success: bool = True) -> Job:
        spec = POLICIES.get(name)
        if spec is None:
            raise ValueError(f"Unknown policy {name!r}; available: {sorted(POLICIES)}")
        status = self.status()
        if status.mode == "real" and not spec.real_ok:
            raise ValueError(f"{name} is not validated for the real arm (trained on another setup); run it in simulation")
        loaded = self.load_policy(name)
        missing = [loaded.camera_for(k) for k in loaded.image_keys if loaded.camera_for(k) not in status.cameras]
        if missing:
            raise ValueError(f"{name} needs cameras {missing}; available: {list(status.cameras)}")

        def make(limits: JointLimits) -> Behavior:
            return PolicyBehavior(
                loaded, self.backend.render, self.backend.task_signals, seconds, self.backend.fps,
                stop_on_success=stop_on_success,
            )

        return self._call("start_job", ("policy", {"name": name, "seconds": seconds}, make))

    def place_object(self, name: str, x: float, y: float) -> dict[str, np.ndarray]:
        """Move a sim object; returns once a status snapshot taken after the move is published."""
        positions = self._call("place_object", (name, x, y))
        tick = self.status().tick
        deadline = time.monotonic() + 1.0
        while self.status().tick <= tick + 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        return positions

    def stop(self) -> Job | None:
        """Cancel the active job; takes effect at the next tick (<= one control period)."""
        return self._call("stop", ())

    def reset(self, seed: int | None) -> None:
        self._call("reset", (seed,), timeout=30.0)

    def render(self, cameras: list[str]) -> dict[str, np.ndarray]:
        return self._call("render", (cameras,))

    def _call(self, name: str, args: tuple, timeout: float = 10.0):
        if not self._thread.is_alive():
            raise RuntimeError("Control thread is not running")
        future: Future = Future()
        self._commands.put((name, args, future))
        return future.result(timeout)

    # ---------- control thread ----------
    def _run(self) -> None:
        try:
            backend = self._factory()
            backend.connect()
            if self._viewer:
                backend.attach_viewer()
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready.set()
            return

        self.backend = backend
        self.limits = backend.joint_limits()
        try:
            rest = np.clip(backend.rest_pose(), self.limits.lower, self.limits.upper)
            table_z = REAL_TABLE_TOP_Z if backend.mode == "real" else TABLE_TOP_Z
            self.planner = MotionPlanner(ArmKinematics(table_top_z=table_z))
            guard = TableGuard(rest_pose=rest, table_top_z=table_z) if self._use_table_guard else None
        except BaseException as exc:  # noqa: BLE001
            backend.disconnect()
            self._startup_error = exc
            self._ready.set()
            return
        self.safety = SafetyLimiter(self.limits, backend.fps, self._speed, guard)
        measured = backend.read_joints()
        self.safety.resync(measured)
        self._active: tuple[Job, Behavior, int] | None = None
        self._guard_holds = 0
        self._tick = 0
        self._overruns = 0
        self._hz = float(backend.fps)
        self._task = backend.task_signals()
        self._objects = backend.object_positions()
        self._publish(measured)
        self._ready.set()

        period = 1.0 / backend.fps
        next_tick = time.perf_counter()
        last_tick_start = next_tick
        try:
            while not self._shutdown.is_set():
                tick_start = time.perf_counter()
                dt = tick_start - last_tick_start
                if dt > 0:
                    self._hz = 0.95 * self._hz + 0.05 * (1.0 / dt)
                last_tick_start = tick_start

                self._drain_commands()
                measured = backend.read_joints()
                contact = backend.contact_faults()
                if contact:
                    for message in contact:
                        self._fault(message)
                    self._fail_active("; ".join(contact))
                    # Stop pushing: the new hold target is wherever the arm actually is.
                    self.safety.resync(measured)
                target = self._behavior_target(measured)
                try:
                    safe, notes = self.safety.apply(target, measured)
                except UnsafeTargetError as exc:
                    safe, notes = self.safety.last_command, []
                    if isinstance(exc, TableGuardError) and self._active and self._active[1].hold_on_table_guard:
                        self._guard_holds += 1
                        notes = ["table_guard_held"]
                    else:
                        self._fail_active(str(exc))
                if notes and self._active is not None:
                    job = self._active[0]
                    for note in notes:
                        if note not in job.notes:
                            job.notes.append(note)
                backend.command(safe)
                backend.sync_viewer()
                self._tick += 1
                if self._tick % 10 == 0:
                    self._task = backend.task_signals()
                    self._objects = backend.object_positions()
                self._publish(backend.read_joints())

                if self.realtime:
                    next_tick += period
                    sleep = next_tick - time.perf_counter()
                    if sleep > 0:
                        time.sleep(sleep)
                    elif sleep < -period:
                        self._overruns += 1
                        next_tick = time.perf_counter()
        except BaseException as exc:  # noqa: BLE001
            log.exception("Control loop crashed")
            self._fault(f"control loop crashed: {exc!r}")
            self._fail_active(f"control loop crashed: {exc!r}")
        finally:
            self._drain_commands(closing=True)
            backend.disconnect()
            with self._status_lock:
                if self._status is not None:
                    self._status.connected = False

    def _behavior_target(self, measured: np.ndarray) -> np.ndarray:
        hold = self.safety.last_command
        if self._active is None:
            return hold
        job, behavior, tick0 = self._active
        if job.state == "accepted":
            # Start from the commanded pose, not the measured one: on the real arm gravity sags joints a
            # few degrees below their goal, and starting each job from the sagged pose makes the arm creep.
            if self.safety.table_guard is not None:
                self.safety.table_guard.set_clearance(behavior.table_clearance_m)
            self._guard_holds = 0
            behavior.start(hold.copy())
            job.state = "running"
            job.started_at = time.time()
            job.log("running")
        t = (self._tick - tick0) / self.backend.fps
        if t > self.max_job_seconds:
            self._fail_active(f"exceeded max job duration {self.max_job_seconds}s")
            return hold
        target = behavior.step(measured, t)
        job.progress = round(behavior.progress(t), 3)
        job.metrics = behavior.metrics()
        if behavior.hold_on_table_guard:
            job.metrics["table_guard_holds"] = float(self._guard_holds)
        if target is None:
            outcome = behavior.outcome(measured)
            job.reached = outcome.reached
            job.final_error = outcome.final_error
            job.notes.extend(n for n in outcome.notes if n not in job.notes)
            if outcome.error:
                self._fail_active(outcome.error, fault=False)
                return hold
            self._finish_active("completed", "completed" + ("" if outcome.reached else " (target not reached)"))
            if not outcome.reached:
                # Stop pushing into whatever blocked the arm: hold where it actually is.
                job.notes.append("holding_measured_pose")
                self.safety.resync(measured)
                return measured
            return hold
        return target

    def _drain_commands(self, closing: bool = False) -> None:
        while True:
            try:
                name, args, future = self._commands.get_nowait()
            except queue.Empty:
                return
            if closing:
                future.set_exception(RuntimeError("Runtime is shutting down"))
                continue
            try:
                future.set_result(getattr(self, f"_cmd_{name}")(*args))
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)

    def _cmd_start_job(self, kind: str, params: dict, make: Callable[[JointLimits], Behavior]) -> Job:
        if self._active is not None:
            busy, behavior, _ = self._active
            if not behavior.preemptible:
                raise MotionBusyError(f"Arm is busy with {busy.id} ({busy.kind}); call stop or wait for it to finish")
            self._cancel_active(f"preempted by {kind}")
        job = self.jobs.create(kind, params)
        self._active = (job, make(self.limits), self._tick)
        return job

    def _cmd_stop(self) -> Job | None:
        return self._cancel_active("stop requested")

    def _cmd_reset(self, seed: int | None) -> None:
        if self._active is not None:
            raise MotionBusyError(f"Cannot reset while {self._active[0].id} is running; call stop first")
        self.backend.reset(seed)
        self.safety.resync(self.backend.read_joints())
        self._task = self.backend.task_signals()
        self._objects = self.backend.object_positions()

    def _cmd_at_rest(self) -> bool:
        return self.backend.at_rest()

    def _cmd_place_object(self, name: str, x: float, y: float) -> dict[str, np.ndarray]:
        self.backend.place_object(name, x, y)
        self._objects = self.backend.object_positions()
        return self._objects

    def _cmd_render(self, cameras: list[str]) -> dict[str, np.ndarray]:
        return {name: self.backend.render(name) for name in cameras}

    def _cmd_snapshot_raw(self) -> tuple:
        measured = self.backend.read_joints()
        frames = {name: self.backend.render(name) for name in self.backend.camera_names}
        geometry = {name: value for name in frames if (value := self.backend.camera_geometry(name)) is not None}
        return measured.copy(), frames, list(self._faults), self.backend.mode, geometry

    def _cancel_active(self, reason: str) -> Job | None:
        if self._active is None:
            return None
        job = self._active[0]
        self._finish_active("cancelled", f"cancelled: {reason}")
        return job

    def _fail_active(self, reason: str, fault: bool = True) -> None:
        if self._active is None:
            return
        self._active[0].error = reason
        if fault:
            self._fault(reason)
        self._finish_active("failed", f"failed: {reason}")

    def _finish_active(self, state: str, message: str) -> None:
        job, behavior, _ = self._active
        job.state = state  # type: ignore[assignment]
        job.ended_at = time.time()
        job.log(message)
        self._active = None
        if self.safety.table_guard is not None:
            self.safety.table_guard.set_clearance(None)
        try:
            behavior.finish()
        except Exception:  # noqa: BLE001
            log.exception("Behaviour cleanup failed")

    def _fault(self, message: str) -> None:
        self._faults.append(f"{time.strftime('%H:%M:%S')} {message}")
        print(f"[so101-runtime] fault: {message}", flush=True)

    def _publish(self, measured: np.ndarray) -> None:
        status = RuntimeStatus(
            mode=self.backend.mode,
            backend=self.backend.name,
            connected=True,
            fps=self.backend.fps,
            measured_hz=round(self._hz, 1),
            overruns=self._overruns,
            tick=self._tick,
            sim_time=self.backend.sim_time(),
            joints=measured.copy(),
            commanded=self.safety.last_command.copy(),
            limits=self.limits,
            cameras=self.backend.camera_names,
            active_job_id=self._active[0].id if self._active else None,
            task=self._task,
            faults=list(self._faults),
            objects=dict(self._objects),
        )
        with self._status_lock:
            self._status = status
