"""Job records: every motion the agent starts is a job with an id, state, events and outcome."""

from __future__ import annotations

import itertools
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

JobState = Literal["accepted", "running", "completed", "failed", "cancelled"]
TERMINAL: frozenset[str] = frozenset({"completed", "failed", "cancelled"})


@dataclass
class JobEvent:
    t: float
    message: str


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    state: JobState = "accepted"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    progress: float = 0.0
    events: list[JobEvent] = field(default_factory=list)
    reached: bool | None = None
    final_error: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def log(self, message: str) -> None:
        self.events.append(JobEvent(time.time(), message))

    @property
    def done(self) -> bool:
        return self.state in TERMINAL


class JobStore:
    def __init__(self, keep: int = 100):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self.keep = keep

    def create(self, kind: str, params: dict) -> Job:
        with self._lock:
            job = Job(id=f"job-{next(self._ids)}", kind=kind, params=params)
            job.log("accepted")
            self._jobs[job.id] = job
            while len(self._jobs) > self.keep:
                self._jobs.popitem(last=False)
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, n: int) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())[-n:]
