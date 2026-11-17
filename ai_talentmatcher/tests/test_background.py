"""Rate limiter, error classification, job queue retries, matching retries."""

import json
import threading
import time

import litellm
import pytest

from talentmatcher import config, llm, matching
from talentmatcher.jobs import JobManager, JobStatus
from talentmatcher.llm import LLMError
from talentmatcher.profiles import CandidateProfile, ExtractionError
from talentmatcher.ratelimit import RateLimiter

PROFILE = CandidateProfile(firstName="Alex", lastName="Morgan", phone=None, email=None,
                           location="Rabat", skills=["Python"])


# ---------- rate limiter ----------

class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_limiter_blocks_after_rpm_until_window_slides():
    clock = FakeClock()
    rl = RateLimiter(3, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        assert rl.acquire()
    assert clock.slept == []
    assert rl.acquire()  # 4th request must wait ~60s for the oldest slot to expire
    assert sum(clock.slept) == pytest.approx(60.0)


def test_limiter_pause_blocks_everyone():
    clock = FakeClock()
    rl = RateLimiter(100, clock=clock, sleep=clock.sleep)
    rl.pause(30)
    assert rl.snapshot()["cooldown_seconds"] == 30
    assert rl.acquire()
    assert sum(clock.slept) == pytest.approx(30.0)


def test_limiter_timeout():
    clock = FakeClock()
    rl = RateLimiter(1, clock=clock, sleep=clock.sleep)
    assert rl.acquire()
    assert rl.acquire(timeout=5) is False


# ---------- LLM error classification ----------

def _raise(exc):
    def fn(**kwargs):
        raise exc
    return fn


@pytest.fixture
def fresh_limiter(monkeypatch):
    rl = RateLimiter(0)
    monkeypatch.setattr(llm, "rate_limiter", rl)
    return rl


def test_503_is_retryable(monkeypatch, fresh_limiter):
    exc = litellm.ServiceUnavailableError("high demand", llm_provider="gemini", model="m")
    monkeypatch.setattr(litellm, "completion", _raise(exc))
    with pytest.raises(LLMError) as info:
        llm.chat([{"role": "user", "content": "x"}], temperature=0)
    assert info.value.retryable


def test_429_is_retryable_and_pauses_all_requests(monkeypatch, fresh_limiter):
    body = '{"error": {"code": 429, "details": [{"retryDelay": "37s"}]}}'
    exc = litellm.RateLimitError(body, llm_provider="gemini", model="m")
    monkeypatch.setattr(litellm, "completion", _raise(exc))
    with pytest.raises(LLMError) as info:
        llm.chat([{"role": "user", "content": "x"}], temperature=0)
    assert info.value.retryable and info.value.retry_after == 37
    assert fresh_limiter.snapshot()["cooldown_seconds"] == pytest.approx(37, abs=1)


GEMINI_DAILY_QUOTA = (
    '{"error": {"code": 429, "message": "Quota exceeded for metric: generate_content_free_tier_requests, '
    'limit: 20\\nPlease retry in 15h13m53.511185372s.", "status": "RESOURCE_EXHAUSTED", '
    '"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "54833s"}]}}'
)


@pytest.mark.parametrize("text, seconds", [
    ('"retryDelay": "54833s"', 54833),
    ("Please retry in 15h13m53.5s.", 15 * 3600 + 13 * 60 + 53.5),
    ("Please retry in 37.2s.", 37.2),
    ("no hint here", None),
])
def test_retry_after_parsing(text, seconds):
    assert llm._retry_after_seconds(Exception(text)) == (pytest.approx(seconds) if seconds else None)


def test_daily_quota_fails_fast_and_blocks_later_calls(monkeypatch, fresh_limiter):
    exc = litellm.RateLimitError(GEMINI_DAILY_QUOTA, llm_provider="gemini", model="m")
    calls = []

    def completion(**kwargs):
        calls.append(1)
        raise exc

    monkeypatch.setattr(litellm, "completion", completion)
    for _ in range(2):
        with pytest.raises(LLMError) as info:
            llm.chat([{"role": "user", "content": "x"}], temperature=0)
        assert not info.value.retryable and "exhausted" in str(info.value) and "15h" in str(info.value)
    assert len(calls) == 1  # the second call never reached the provider


def test_bad_key_is_not_retryable(monkeypatch, fresh_limiter):
    exc = litellm.AuthenticationError("invalid key", llm_provider="gemini", model="m")
    monkeypatch.setattr(litellm, "completion", _raise(exc))
    with pytest.raises(LLMError) as info:
        llm.chat([{"role": "user", "content": "x"}], temperature=0)
    assert not info.value.retryable


# ---------- job queue ----------

class ScriptedExtract:
    """Fake extraction step: raises/returns scripted outcomes in order."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, cv_text):
        with self.lock:
            self.calls += 1
            outcome = self.outcomes.pop(0) if self.outcomes else (PROFILE, [])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def make_manager():
    managers = []

    def factory(extract, **kwargs):
        kwargs.setdefault("backoff_base", 0.02)
        kwargs.setdefault("backoff_max", 0.1)
        m = JobManager(extract, **kwargs)
        m.start()
        managers.append(m)
        return m

    yield factory
    for m in managers:
        m.stop()


def test_job_is_retried_after_temporary_errors(make_manager):
    extract = ScriptedExtract(
        LLMError("503", retryable=True),
        LLMError("429", retryable=True, retry_after=0.05),
        (PROFILE, []),
    )
    m = make_manager(extract, workers=1, max_attempts=5)
    outcome, job, _ = m.submit("a.pdf", "hash-a", "text")
    assert outcome == "queued"

    assert wait_for(lambda: job.status == JobStatus.DONE)
    assert job.attempts == 3 and extract.calls == 3
    [cand] = m.list_candidates()
    assert cand.profile.firstName == "Alex"
    assert "3 attempts" in cand.warnings[0]


def test_job_fails_after_max_attempts(make_manager):
    extract = ScriptedExtract(*[LLMError("503", retryable=True)] * 10)
    m = make_manager(extract, workers=1, max_attempts=3)
    _, job, _ = m.submit("a.pdf", "hash-a", "text")
    assert wait_for(lambda: job.status == JobStatus.FAILED)
    assert job.attempts == 3 and "Stopped after 3 attempts" in job.error
    assert m.list_candidates() == []


def test_permanent_errors_are_not_retried(make_manager):
    extract = ScriptedExtract(LLMError("bad key", retryable=False), ExtractionError("invalid JSON"))
    m = make_manager(extract, workers=1, max_attempts=5)
    _, j1, _ = m.submit("a.pdf", "hash-a", "text")
    _, j2, _ = m.submit("b.pdf", "hash-b", "text")
    assert wait_for(lambda: j1.status == j2.status == JobStatus.FAILED)
    assert j1.attempts == j2.attempts == 1


def test_waiting_job_does_not_block_others(make_manager):
    # a.pdf must wait a long time before retrying; b.pdf should still finish right away.
    extract = ScriptedExtract(LLMError("429", retryable=True, retry_after=60), (PROFILE, []))
    m = make_manager(extract, workers=1, backoff_max=60)
    _, a, _ = m.submit("a.pdf", "hash-a", "text")
    assert wait_for(lambda: a.status == JobStatus.WAITING_RETRY)
    _, b, _ = m.submit("b.pdf", "hash-b", "text")
    assert wait_for(lambda: b.status == JobStatus.DONE, timeout=2)
    assert a.status == JobStatus.WAITING_RETRY and m.retry_in(a) > 30


def test_same_file_is_cached(make_manager):
    extract = ScriptedExtract()
    m = make_manager(extract, workers=1)
    _, job, _ = m.submit("a.pdf", "hash-a", "text")
    assert m.submit("copy.pdf", "hash-a", "text")[0] in ("already_queued", "cached")
    assert wait_for(lambda: job.status == JobStatus.DONE)
    assert m.submit("again.pdf", "hash-a", "text")[0] == "cached"
    assert extract.calls == 1


def test_failed_job_can_be_retried_manually(make_manager):
    extract = ScriptedExtract(LLMError("bad key", retryable=False), (PROFILE, []))
    m = make_manager(extract, workers=1)
    _, job, _ = m.submit("a.pdf", "hash-a", "text")
    assert wait_for(lambda: job.status == JobStatus.FAILED)
    m.retry(job.id)
    assert wait_for(lambda: job.status == JobStatus.DONE)


def test_clear_discards_in_flight_results(make_manager):
    gate = threading.Event()

    def slow_extract(text):
        gate.wait(2)
        return PROFILE, []

    m = make_manager(slow_extract, workers=1)
    _, job, _ = m.submit("a.pdf", "hash-a", "text")
    assert wait_for(lambda: job.status == JobStatus.PROCESSING)
    m.clear()
    gate.set()
    time.sleep(0.1)
    assert m.list_candidates() == [] and m.list_jobs() == []


# ---------- matching retries ----------

def test_matching_retries_temporary_errors(monkeypatch):
    answers = [LLMError("503", retryable=True), json.dumps({"required_skills": ["Python"], "constraints": [], "assessments": [{"id": "C1", "location_check": "no constraint", "meets_constraints": True, "matching_skills": ["Python"], "missing_skills": []}], "no_match_reason": None})]

    def fake_chat(messages, **kwargs):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(matching, "chat", fake_chat)
    sleeps = []
    assert "**Alex Morgan (C1)**" in matching.match_candidates("Python", [PROFILE], sleep=sleeps.append)
    assert len(sleeps) == 1


def test_matching_does_not_retry_permanent_errors(monkeypatch):
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(1)
        raise LLMError("bad key", retryable=False)

    monkeypatch.setattr(matching, "chat", fake_chat)
    with pytest.raises(LLMError):
        matching.match_candidates("Python", [PROFILE], sleep=lambda s: None)
    assert len(calls) == 1


# ---------- attempt limit ----------

@pytest.mark.parametrize("error", [
    LLMError("The LLM did not answer within 120s.", retryable=True),          # timeout
    LLMError("The model is temporarily unavailable.", retryable=True),        # 503
    LLMError("Rate limit reached; waiting 1s.", retryable=True, retry_after=0.01),  # 429
])
def test_stops_after_5_attempts_keeps_last_error_and_can_be_retried(make_manager, error):
    extract = ScriptedExtract(*[error] * 5)
    m = make_manager(extract, workers=1, max_attempts=5)
    _, job, _ = m.submit("slow.pdf", "hash-slow", "text")

    assert wait_for(lambda: job.status == JobStatus.FAILED)
    assert job.attempts == 5 and extract.calls == 5
    assert job.error == f"Stopped after 5 attempts. Last error: {error}"

    # Retry keeps the last details while the new run is pending, then clears them on success.
    gate = threading.Event()
    extract.outcomes = []
    original = m._extract

    def gated(text):
        gate.wait(2)
        return original(text)

    m._extract = gated
    m.retry(job.id)
    assert wait_for(lambda: job.status == JobStatus.PROCESSING)
    assert job.attempts == 1 and "Stopped after 5 attempts" in job.error
    gate.set()
    assert wait_for(lambda: job.status == JobStatus.DONE)
    assert job.error is None


def test_default_limit_is_5():
    assert config.JOB_MAX_ATTEMPTS == 5
    assert JobManager(lambda t: (PROFILE, []))._max_attempts == 5


def test_litellm_internal_retries_are_disabled_and_output_is_capped(monkeypatch, fresh_limiter):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "hi"}}])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    llm.chat([{"role": "user", "content": "x"}], temperature=0, max_tokens=123)
    assert seen["max_retries"] == 0 and seen["max_tokens"] == 123


def test_timeout_error_is_flagged(monkeypatch, fresh_limiter):
    monkeypatch.setattr(litellm, "completion", _raise(litellm.Timeout("slow", model="m", llm_provider="openai")))
    with pytest.raises(LLMError) as info:
        llm.chat([{"role": "user", "content": "x"}], temperature=0)
    assert info.value.retryable
