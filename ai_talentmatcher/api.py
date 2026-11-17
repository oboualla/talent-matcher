"""HTTP API: accepts CV uploads, processes them in the background, serves profiles and matches.

Run a SINGLE process (uvicorn default): jobs and candidates live in this process's memory.
Interactive docs: http://localhost:8000/docs
"""

import hashlib
import logging
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from talentmatcher import config
from talentmatcher.jobs import Candidate, Job, JobManager
from talentmatcher.llm import LLMError, rate_limiter
from talentmatcher.matching import MatchingInputError, match_candidates
from talentmatcher.pdf_text import PdfTextError, extract_text
from talentmatcher.profiles import extract_profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

manager = JobManager(
    extract_profile,
    workers=config.WORKER_CONCURRENCY,
    max_attempts=config.JOB_MAX_ATTEMPTS,
    backoff_base=config.RETRY_BASE_SECONDS,
    backoff_max=config.RETRY_MAX_SECONDS,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.start()
    yield
    manager.stop()


app = FastAPI(title="AI_TalentMatcher API", lifespan=lifespan)


# ---------- response shapes ----------

class JobOut(BaseModel):
    id: str
    filename: str
    status: str
    attempts: int
    retry_in_seconds: Optional[float]
    error: Optional[str]
    candidate_id: Optional[str]


class CandidateOut(BaseModel):
    id: str
    filename: str
    profile: dict
    warnings: list[str]
    cv_text: str


class UploadResult(BaseModel):
    filename: str
    outcome: str  # queued | already_queued | cached | rejected
    message: str
    job_id: Optional[str] = None
    candidate_id: Optional[str] = None


class MatchIn(BaseModel):
    description: str


class MatchOut(BaseModel):
    answer: str
    candidate_count: int


def _job_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        filename=job.filename,
        status=job.status.value,
        attempts=job.attempts,
        retry_in_seconds=None if (r := manager.retry_in(job)) is None else round(r, 1),
        error=job.error,
        candidate_id=job.candidate_id,
    )


def _candidate_out(c: Candidate) -> CandidateOut:
    return CandidateOut(
        id=c.id, filename=c.filename, profile=c.profile.model_dump(), warnings=c.warnings, cv_text=c.cv_text
    )


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    return {
        "model": config.LLM_MODEL,
        # Only the host, never the key.
        "endpoint": urlparse(config.LLM_API_BASE).netloc if config.LLM_API_BASE else "provider default",
        "json_mode": config.LLM_JSON_MODE,
        "workers": config.WORKER_CONCURRENCY,
        "max_attempts": config.JOB_MAX_ATTEMPTS,
        "rate_limit": rate_limiter.snapshot(),
        "jobs": manager.counts(),
    }


@app.post("/cvs", status_code=202, response_model=list[UploadResult])
async def upload_cvs(files: list[UploadFile] = File(...)):
    """Accept PDFs and answer immediately. PDF text extraction is fast and done here, so
    unreadable files are rejected at once; the slow LLM step is queued in the background."""
    results = []
    max_bytes = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    for f in files:
        name = f.filename or "unnamed.pdf"
        data = await f.read()
        if len(data) > max_bytes:
            results.append(UploadResult(filename=name, outcome="rejected",
                                        message=f"File is larger than {config.MAX_UPLOAD_MB:g} MB."))
            continue
        try:
            text = extract_text(data)
        except PdfTextError as exc:
            results.append(UploadResult(filename=name, outcome="rejected", message=str(exc)))
            continue

        outcome, job, cand = manager.submit(name, hashlib.sha256(data).hexdigest(), text)
        message = {
            "queued": "Accepted: the profile is being extracted in the background.",
            "already_queued": "This file is already being processed.",
            "cached": "This file was already extracted; the existing profile is reused (no LLM call).",
        }[outcome]
        results.append(UploadResult(
            filename=name, outcome=outcome, message=message,
            job_id=job.id if job else None, candidate_id=cand.id if cand else None,
        ))
    return results


@app.get("/jobs", response_model=list[JobOut])
def list_jobs():
    return [_job_out(j) for j in manager.list_jobs()]


@app.post("/jobs/{job_id}/retry", response_model=JobOut)
def retry_job(job_id: str):
    job = manager.retry(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return _job_out(job)


@app.get("/candidates", response_model=list[CandidateOut])
def list_candidates():
    return [_candidate_out(c) for c in manager.list_candidates()]


@app.delete("/candidates/{candidate_id}", status_code=204)
def delete_candidate(candidate_id: str):
    if not manager.delete_candidate(candidate_id):
        raise HTTPException(404, "Candidate not found.")


@app.delete("/candidates", status_code=204)
def clear_all():
    manager.clear()


@app.post("/match", response_model=MatchOut)
def match(body: MatchIn):
    # Sync endpoint: FastAPI runs it in a thread pool, so waiting on the rate limiter
    # or a retry does not block the other endpoints (uploads, polling).
    profiles = [c.profile for c in manager.list_candidates()]
    try:
        answer = match_candidates(body.description, profiles)
    except MatchingInputError as exc:
        raise HTTPException(422, str(exc))
    except LLMError as exc:
        raise HTTPException(503 if exc.retryable else 502, str(exc))
    return MatchOut(answer=answer, candidate_count=len(profiles))
