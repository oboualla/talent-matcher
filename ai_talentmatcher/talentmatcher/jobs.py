"""In-memory job queue that turns uploaded CVs into candidate profiles in the background.

Life of a job:

    submit ──► QUEUED ──► PROCESSING ──► DONE        (profile stored as a candidate)
                  ▲            │
                  │            ├──► WAITING_RETRY ──(after backoff)──► QUEUED
                  │            │      temporary error: 429, 503, timeout, network
                  │            └──► FAILED       permanent error, or too many attempts
                  └── retry() from the UI for a FAILED job

Scheduling uses a min-heap ordered by "ready at" time, so a job waiting for its
retry doesn't block other jobs. All state lives in memory (lost on restart).
"""

import heapq
import itertools
import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .llm import LLMError
from .profiles import CandidateProfile, ExtractionError


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    WAITING_RETRY = "waiting_retry"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


log = logging.getLogger("talentmatcher.jobs")

ACTIVE = {JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.WAITING_RETRY}


@dataclass
class Job:
    id: str
    filename: str
    sha256: str
    cv_text: str
    status: JobStatus = JobStatus.QUEUED
    attempts: int = 0
    ready_at: float = 0.0  # monotonic time when the job may run
    error: Optional[str] = None
    candidate_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Candidate:
    id: str
    filename: str
    sha256: str
    profile: CandidateProfile
    cv_text: str
    warnings: list[str]
    created_at: float = field(default_factory=time.time)


# Signature of the extraction step: cv_text -> (profile, warnings)
ExtractFn = Callable[[str], tuple[CandidateProfile, list[str]]]


class JobManager:
    def __init__(
        self,
        extract_fn: ExtractFn,
        *,
        workers: int = 2,
        max_attempts: int = 5,
        backoff_base: float = 5.0,
        backoff_max: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._extract = extract_fn
        self._workers_count = max(1, workers)
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._clock = clock

        self._cond = threading.Condition()  # protects everything below
        self._jobs: dict[str, Job] = {}
        self._candidates: dict[str, Candidate] = {}
        self._heap: list[tuple[float, int, str]] = []  # (ready_at, tie-breaker, job_id)
        self._seq = itertools.count()
        self._threads: list[threading.Thread] = []
        self._stopping = False

    # ---------- lifecycle ----------

    def start(self) -> None:
        with self._cond:
            self._stopping = False
        for i in range(self._workers_count):
            t = threading.Thread(target=self._worker_loop, name=f"cv-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout)
        self._threads.clear()

    # ---------- public API (thread-safe) ----------

    def submit(self, filename: str, sha256: str, cv_text: str) -> tuple[str, Optional[Job], Optional[Candidate]]:
        """Queue a CV, unless it is cached. Returns (outcome, job, candidate) where outcome is
        "queued", "cached" (already extracted) or "already_queued"."""
        with self._cond:
            for cand in self._candidates.values():
                if cand.sha256 == sha256:
                    return "cached", None, cand
            for job in self._jobs.values():
                if job.sha256 == sha256 and job.status in ACTIVE:
                    return "already_queued", job, None

            job = Job(id=uuid.uuid4().hex[:12], filename=filename, sha256=sha256, cv_text=cv_text)
            self._jobs[job.id] = job
            self._schedule(job, delay=0)
            return "queued", job, None

    def retry(self, job_id: str) -> Optional[Job]:
        """Re-queue a FAILED or CANCELLED job with a fresh attempt budget.

        The last error is kept (shown as "previous attempt failed") until the new run
        succeeds or fails, so the details are never lost.
        """
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None or job.status in ACTIVE or job.status == JobStatus.DONE:
                return job
            job.attempts = 0
            self._schedule(job, delay=0)
            return job

    def list_jobs(self) -> list[Job]:
        with self._cond:
            return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def list_candidates(self) -> list[Candidate]:
        with self._cond:
            return sorted(self._candidates.values(), key=lambda c: c.created_at)

    def delete_candidate(self, candidate_id: str) -> bool:
        with self._cond:
            cand = self._candidates.pop(candidate_id, None)
            if cand:
                # Forget the finished job too, so the same file can be uploaded again.
                for job_id, job in list(self._jobs.items()):
                    if job.candidate_id == candidate_id:
                        del self._jobs[job_id]
            return cand is not None

    def clear(self) -> None:
        """Drop all candidates and jobs; jobs currently being processed are discarded."""
        with self._cond:
            for job in self._jobs.values():
                if job.status in ACTIVE:
                    job.status = JobStatus.CANCELLED
            self._jobs.clear()
            self._candidates.clear()
            self._heap.clear()

    def retry_in(self, job: Job) -> Optional[float]:
        if job.status != JobStatus.WAITING_RETRY:
            return None
        return max(0.0, job.ready_at - self._clock())

    def counts(self) -> dict:
        with self._cond:
            counts = {s.value: 0 for s in JobStatus}
            for job in self._jobs.values():
                counts[job.status.value] += 1
            counts["candidates"] = len(self._candidates)
            return counts

    # ---------- internals ----------

    def _schedule(self, job: Job, delay: float) -> None:
        """Caller holds the lock."""
        job.ready_at = self._clock() + delay
        job.status = JobStatus.WAITING_RETRY if delay > 0 else JobStatus.QUEUED
        job.updated_at = time.time()
        heapq.heappush(self._heap, (job.ready_at, next(self._seq), job.id))
        self._cond.notify()

    def _backoff(self, attempts: int, retry_after: Optional[float]) -> float:
        exponential = self._backoff_base * 2 ** (attempts - 1)
        delay = min(max(exponential, retry_after or 0.0), self._backoff_max)
        # Jitter spreads retries so all workers don't hit the provider at the same instant.
        return delay * random.uniform(1.0, 1.25)

    def _next_ready_job(self) -> Optional[Job]:
        """Block until a job is ready to run (or we are stopping). Holds the lock."""
        with self._cond:
            while not self._stopping:
                if not self._heap:
                    self._cond.wait()
                    continue
                ready_at, _, job_id = self._heap[0]
                job = self._jobs.get(job_id)
                # Stale heap entry: job deleted, cancelled, or rescheduled with a new ready_at.
                if job is None or job.status not in (JobStatus.QUEUED, JobStatus.WAITING_RETRY) or job.ready_at != ready_at:
                    heapq.heappop(self._heap)
                    continue
                wait = ready_at - self._clock()
                if wait > 0:
                    self._cond.wait(timeout=wait)  # woken early if a sooner job arrives
                    continue
                heapq.heappop(self._heap)
                job.status = JobStatus.PROCESSING
                job.attempts += 1
                job.updated_at = time.time()
                return job
            return None

    def _worker_loop(self) -> None:
        while True:
            job = self._next_ready_job()
            if job is None:
                return
            try:
                profile, warnings = self._extract(job.cv_text)  # LLM work, outside the lock
            except LLMError as exc:
                self._on_failure(job, str(exc), exc.retryable, exc.retry_after)
            except ExtractionError as exc:
                self._on_failure(job, str(exc), retryable=False, retry_after=None)
            except Exception as exc:  # a bug must not kill the worker thread
                self._on_failure(job, f"Unexpected error: {type(exc).__name__}: {exc}", False, None)
            else:
                self._on_success(job, profile, warnings)

    def _on_success(self, job: Job, profile: CandidateProfile, warnings: list[str]) -> None:
        with self._cond:
            if self._jobs.get(job.id) is not job:  # cleared while we were working
                return
            if job.attempts > 1:
                warnings = [f"Succeeded after {job.attempts} attempts (temporary provider errors)."] + warnings
            cand = Candidate(
                id=uuid.uuid4().hex[:12],
                filename=job.filename,
                sha256=job.sha256,
                profile=profile,
                cv_text=job.cv_text,
                warnings=warnings,
            )
            self._candidates[cand.id] = cand
            job.status = JobStatus.DONE
            log.info("%s: profile extracted (attempt %d)", job.filename, job.attempts)
            job.candidate_id = cand.id
            job.error = None
            job.updated_at = time.time()

    def _on_failure(self, job: Job, message: str, retryable: bool, retry_after: Optional[float]) -> None:
        with self._cond:
            if self._jobs.get(job.id) is not job:
                return
            if retryable and job.attempts < self._max_attempts:
                job.error = message
                delay = self._backoff(job.attempts, retry_after)
                log.warning("%s: attempt %d/%d failed (%s); retrying in %.0fs",
                            job.filename, job.attempts, self._max_attempts, message, delay)
                self._schedule(job, delay=delay)
                return

            # Final state: keep the details of the last attempt for the UI.
            if retryable:
                job.error = f"Stopped after {job.attempts} attempts. Last error: {message}"
            else:
                job.error = message
            log.error("%s: failed after %d attempt(s): %s", job.filename, job.attempts, message)
            job.status = JobStatus.FAILED
            job.updated_at = time.time()
