"""End-to-end API flow with the LLM mocked: upload -> 202 -> poll -> profile -> match."""

import json
import time

import pymupdf
import pytest
from fastapi.testclient import TestClient

import api
from talentmatcher import matching
from talentmatcher.jobs import JobManager
from talentmatcher.llm import LLMError
from talentmatcher.profiles import CandidateProfile

PROFILE = CandidateProfile(firstName="Alex", lastName="Morgan", phone=None, email=None,
                           location="Rabat", skills=["Python", "SQL"])


def make_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((50, 72), text, fontsize=11)
    return doc.tobytes()


@pytest.fixture
def client(monkeypatch):
    attempts = {"n": 0}

    def flaky_extract(text):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LLMError("The model is temporarily unavailable (high demand).", retryable=True)
        return PROFILE, []

    manager = JobManager(flaky_extract, workers=1, backoff_base=0.02, backoff_max=0.05)
    monkeypatch.setattr(api, "manager", manager)
    with TestClient(api.app) as c:  # runs lifespan: starts/stops the workers
        yield c


def poll_until_idle(client, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = client.get("/jobs").json()
        if jobs and all(j["status"] in ("done", "failed") for j in jobs):
            return jobs
        time.sleep(0.02)
    raise AssertionError(f"jobs still pending: {jobs}")


def test_upload_is_accepted_immediately_then_processed(client, monkeypatch):
    cv = make_pdf("Alex Morgan - Rabat - Python, SQL developer with dashboards experience")
    resp = client.post("/cvs", files=[
        ("files", ("alex.pdf", cv, "application/pdf")),
        ("files", ("broken.pdf", b"not a pdf", "application/pdf")),
    ])
    assert resp.status_code == 202
    outcomes = {r["filename"]: r["outcome"] for r in resp.json()}
    assert outcomes == {"alex.pdf": "queued", "broken.pdf": "rejected"}

    [job] = poll_until_idle(client)
    assert job["status"] == "done" and job["attempts"] == 2  # first try hit a fake 503

    [cand] = client.get("/candidates").json()
    assert cand["profile"]["firstName"] == "Alex"

    # Same file again: served from the cache, no new job.
    again = client.post("/cvs", files=[("files", ("copy.pdf", cv, "application/pdf"))]).json()
    assert again[0]["outcome"] == "cached"

    answer = json.dumps({"required_skills": ["Python", "SQL"], "constraints": [], "assessments": [
        {"id": "C1", "location_check": "no constraint", "meets_constraints": True,
         "matching_skills": ["Python", "SQL"], "missing_skills": []}], "no_match_reason": None})
    monkeypatch.setattr(matching, "chat", lambda messages, **kw: answer)
    match = client.post("/match", json={"description": "Python dashboard"})
    assert match.status_code == 200 and match.json()["candidate_count"] == 1


def test_match_errors(client, monkeypatch):
    assert client.post("/match", json={"description": "Python"}).status_code == 422  # no candidates

    client.post("/cvs", files=[("files", ("alex.pdf", make_pdf("Alex Morgan, Python developer in Rabat"), "application/pdf"))])
    poll_until_idle(client)
    assert client.post("/match", json={"description": "  "}).status_code == 422

    def down(messages, **kw):
        raise LLMError("bad key", retryable=False)

    monkeypatch.setattr(matching, "chat", down)
    resp = client.post("/match", json={"description": "Python"})
    assert resp.status_code == 502 and "bad key" in resp.json()["detail"]


def test_clear_all(client):
    client.post("/cvs", files=[("files", ("alex.pdf", make_pdf("Alex Morgan, Python developer in Rabat"), "application/pdf"))])
    poll_until_idle(client)
    assert client.delete("/candidates").status_code == 204
    assert client.get("/candidates").json() == [] and client.get("/jobs").json() == []
