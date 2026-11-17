"""Thin wrapper around LiteLLM: one function to send messages and get text back.

- Every call first waits for the shared rate limiter (requests per minute).
- Every failure becomes an LLMError with a user-facing message and a `retryable`
  flag, so the job queue knows whether trying again later makes sense:
    retryable:     429 rate limit / quota, 503 overloaded, 5xx, timeouts, network errors
    not retryable: bad API key, unknown model, rejected request, empty answer
"""

import re
from typing import Optional

import litellm

from . import config
from .ratelimit import RateLimiter

# Don't print LiteLLM's "Give Feedback / Get Help" banner on every error.
litellm.suppress_debug_info = True

rate_limiter = RateLimiter(config.LLM_REQUESTS_PER_MINUTE)


class LLMError(Exception):
    """The LLM request failed; str(error) is a user-facing explanation."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after  # seconds suggested by the provider, if any


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Find how long the provider asks us to wait, from headers or the error body."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = headers.get("retry-after")
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass
    text = str(exc)
    # Gemini puts it in the body: "retryDelay": "54833s" ...
    match = re.search(r'retryDelay"?\s*:\s*"(\d+(?:\.\d+)?)s"', text)
    if match:
        return float(match.group(1))
    # ... and/or "Please retry in 15h13m53.5s" / "retry in 37.2s".
    match = re.search(r"retry in ((?:\d+h)?(?:\d+m(?!s))?(?:\d+(?:\.\d+)?s)?)", text)
    if match and match.group(1):
        parts = re.findall(r"(\d+(?:\.\d+)?)([hms])", match.group(1))
        return sum(float(n) * {"h": 3600, "m": 60, "s": 1}[u] for n, u in parts)
    return None


def _human(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m" if hours else f"{rest // 60}m{rest % 60:02d}s"


def _self_hosted_hint() -> str:
    if not config.LLM_API_BASE:
        return ""
    return (
        f" Is the self-hosted server at {config.LLM_API_BASE} still running? (A Kaggle session "
        "stops when idle or after ~12h, and its tunnel URL changes on every restart.)"
    )


def _quota_exhausted_error(wait: float) -> LLMError:
    # A wait longer than RETRY_MAX_SECONDS is a daily/long quota, not a short burst:
    # retrying in the background would only waste time, so the job fails clearly instead.
    return LLMError(
        f"The provider's quota for model '{config.LLM_MODEL}' is exhausted (e.g. the free-tier "
        f"daily request limit). It resets in about {_human(wait)}. Use another model or API key "
        "in .env and restart, or retry this file later.",
        retryable=False,
        retry_after=wait,
    )


def chat(
    messages: list[dict],
    *,
    temperature: float,
    json_mode: bool = False,
    wait_timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Send one chat request. `wait_timeout` bounds how long we wait for a rate-limit slot."""
    cooldown = rate_limiter.cooldown_remaining()
    if cooldown > config.RETRY_MAX_SECONDS:  # quota known to be exhausted: don't even try
        raise _quota_exhausted_error(cooldown)
    if not rate_limiter.acquire(timeout=wait_timeout):
        raise LLMError(
            "The request budget for this minute is used up (rate limit). Please retry shortly.",
            retryable=True,
        )

    kwargs = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "timeout": config.LLM_TIMEOUT_SECONDS,
        # Retries are decided ONLY by our job queue / matching loop. Without this, the
        # underlying OpenAI client silently retries timeouts twice (3 x timeout per call).
        "max_retries": 0,
    }
    if max_tokens:
        # Caps runaway generations (e.g. JSON mode emitting endless whitespace).
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if config.LLM_API_BASE:
        kwargs["api_base"] = config.LLM_API_BASE
    if config.LLM_API_KEY:
        kwargs["api_key"] = config.LLM_API_KEY

    try:
        response = litellm.completion(**kwargs)
    except litellm.RateLimitError as exc:
        retry_after = _retry_after_seconds(exc) or config.RATE_LIMIT_COOLDOWN_SECONDS
        # The quota belongs to the API key: make *every* worker wait, not just this job.
        rate_limiter.pause(retry_after)
        if retry_after > config.RETRY_MAX_SECONDS:
            raise _quota_exhausted_error(retry_after) from exc
        raise LLMError(
            f"The provider's rate limit or quota was reached; waiting {retry_after:.0f}s.",
            retryable=True,
            retry_after=retry_after,
        ) from exc
    except litellm.ServiceUnavailableError as exc:
        retry_after = _retry_after_seconds(exc)
        raise LLMError(
            "The model is temporarily unavailable (high demand).",
            retryable=True,
            retry_after=None if retry_after is None else min(retry_after, config.RETRY_MAX_SECONDS),
        ) from exc
    except litellm.InternalServerError as exc:
        raise LLMError(f"The model server had an internal error.{_self_hosted_hint()}", retryable=True) from exc
    except litellm.Timeout as exc:
        raise LLMError(
            f"The LLM did not answer within {config.LLM_TIMEOUT_SECONDS:.0f}s.{_self_hosted_hint()}",
            retryable=True,
        ) from exc
    except litellm.APIConnectionError as exc:
        raise LLMError(f"Could not reach the model server.{_self_hosted_hint()}", retryable=True) from exc
    except litellm.AuthenticationError as exc:
        raise LLMError(
            f"Authentication with the LLM provider failed. Check the API key in .env. ({exc})"
        ) from exc
    except litellm.NotFoundError as exc:
        raise LLMError(f"Model '{config.LLM_MODEL}' was not found. Check LLM_MODEL in .env.") from exc
    except litellm.BadRequestError as exc:
        raise LLMError(f"The provider rejected the request: {exc}") from exc
    except Exception as exc:  # unknown failure: don't hammer the provider with retries
        raise LLMError(f"The LLM request failed: {type(exc).__name__}: {exc}") from exc

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        raise LLMError("The LLM returned a response without any message.") from exc

    if not content or not content.strip():
        raise LLMError("The LLM returned an empty answer.")
    return content
