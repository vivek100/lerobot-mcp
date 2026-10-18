"""Run a chunking policy (ACT) on the control clock with asynchronous inference.

The control thread captures an observation (joints + camera frames; sim cameras must render on that
thread) and hands it to a worker thread. While the worker computes the next chunk, the arm keeps
executing the current one. When a chunk arrives, actions that are already in the past (inference
latency) are skipped. Every action still passes through the runtime's SafetyLimiter and table guard.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from so101_mcp.backends.base import TaskSignals
from so101_mcp.policies import LoadedPolicy
from so101_mcp.runtime.behaviors import Behavior, BehaviorOutcome


class PolicyBehavior(Behavior):
    kind = "policy"
    # Picking a 3 cm cube off the table brings the jaw tips to the table top; they may touch, not push in.
    table_clearance_m = 0.0
    hold_on_table_guard = True

    def __init__(
        self,
        loaded: LoadedPolicy,
        render: Callable[[str], np.ndarray],
        task: Callable[[], TaskSignals],
        seconds: float,
        fps: int,
        refill_margin_s: float = 1.0,
        stop_on_success: bool = True,
    ):
        self.loaded = loaded
        self.render = render
        self.task = task
        self.seconds = seconds
        self.fps = fps
        self.refill_margin = int(refill_margin_s * fps)
        self.stop_on_success = stop_on_success
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy")

    def start(self, measured: np.ndarray) -> None:
        self.hold = measured.copy()
        self.tick = 0
        self.chunk: np.ndarray | None = None
        self.chunk_tick = 0  # tick at which the chunk's observation was captured
        self.pending: tuple[Future, int] | None = None
        self.chunks = 0
        self.skipped = 0
        self.success = False
        self.error: str | None = None
        self.reward = 0.0
        self._request(measured)

    def _request(self, measured: np.ndarray) -> None:
        frames = {key: self.render(self.loaded.camera_for(key)) for key in self.loaded.image_keys}
        future = self.executor.submit(self.loaded.predict_chunk, measured.copy(), frames)
        self.pending = (future, self.tick)

    def step(self, measured: np.ndarray, t: float) -> np.ndarray | None:
        if t >= self.seconds or self.error:
            return None
        if self.tick % 10 == 0:
            task = self.task()
            if task.available:
                self.reward = float(task.reward or 0.0)
                self.success = bool(task.success)
                if self.success and self.stop_on_success:
                    return None
        if self.pending is not None and self.pending[0].done():
            future, captured = self.pending
            self.pending = None
            try:
                self.chunk, self.chunk_tick = future.result(), captured
                self.chunks += 1
                self.skipped += min(self.tick - captured, len(self.chunk))
            except Exception as exc:  # noqa: BLE001
                self.error = f"policy inference failed: {exc!r}"
                return None
        target = self.hold
        if self.chunk is not None:
            index = self.tick - self.chunk_tick
            if index < len(self.chunk):
                target = self.chunk[index]
                self.hold = target
            remaining = len(self.chunk) - index
            if self.pending is None and remaining <= self.refill_margin:
                self._request(measured)
        self.tick += 1
        return target

    def progress(self, t: float) -> float:
        return min(t / self.seconds, 1.0)

    def metrics(self) -> dict[str, float]:
        out = {"chunks": float(self.chunks), "stale_actions_skipped": float(self.skipped), "task_reward": self.reward}
        out.update(self.loaded.latency_summary())
        return out

    def outcome(self, measured: np.ndarray) -> BehaviorOutcome:
        notes = [f"{self.chunks} chunks of {self.loaded.chunk_size} actions; inference {self.loaded.latency_summary()}"]
        if self.success:
            notes.append("sim task solved")
        return BehaviorOutcome(reached=self.success, notes=notes, error=self.error)

    def finish(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
