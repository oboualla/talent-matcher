"""Runtime configuration, read from environment variables (see .env.example)."""

import os


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# LiteLLM model string: "<provider>/<model>", e.g. "groq/llama-3.3-70b-versatile"
# or "gemini/gemini-2.0-flash". The provider's API key is read by LiteLLM from
# its standard variable (GROQ_API_KEY, GEMINI_API_KEY, ...).
LLM_MODEL = os.getenv("LLM_MODEL", "groq/llama-3.3-70b-versatile")

# Self-hosted / OpenAI-compatible endpoint (e.g. Ollama on a Kaggle GPU behind a tunnel):
# LLM_MODEL=openai/qwen2.5:7b, LLM_API_BASE=https://<tunnel>/v1, LLM_API_KEY=<proxy token>.
# Leave empty for hosted providers (Groq, Gemini), which use their own key variables.
LLM_API_BASE = os.getenv("LLM_API_BASE", "").strip() or None
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip() or None

# Low temperature for extraction: we want faithful copying, not creativity.
EXTRACTION_TEMPERATURE = float(os.getenv("EXTRACTION_TEMPERATURE", "0"))
MATCHING_TEMPERATURE = float(os.getenv("MATCHING_TEMPERATURE", "0"))

# Ask the provider for its JSON mode (response_format=json_object) when extracting.
LLM_JSON_MODE = _env_bool("LLM_JSON_MODE", True)

LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

# How many times we ask the model to correct an invalid profile before giving up.
EXTRACTION_MAX_ATTEMPTS = int(os.getenv("EXTRACTION_MAX_ATTEMPTS", "3"))

# Upper bound on generated tokens. A profile needs a few hundred; the cap stops runaway
# generations that would otherwise run until the timeout.
EXTRACTION_MAX_TOKENS = int(os.getenv("EXTRACTION_MAX_TOKENS", "1500"))
MATCHING_MAX_TOKENS = int(os.getenv("MATCHING_MAX_TOKENS", "3000"))

# CV text longer than this is truncated before being sent (protects context/quota).
MAX_CV_CHARS = int(os.getenv("MAX_CV_CHARS", "20000"))

# ---- Rate limiting and background processing ----

# Max LLM requests per minute for this API key (all workers + matching combined).
# Set it at or below your provider's free-tier limit. 0 disables the limiter.
LLM_REQUESTS_PER_MINUTE = int(os.getenv("LLM_REQUESTS_PER_MINUTE", "10"))

# Number of background threads extracting CVs in parallel.
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "2"))

# A CV job hitting a temporary error (429/503/timeout/network) is tried at most this many
# times in total; then it is FAILED, keeps its last error, and can be retried from the UI.
JOB_MAX_ATTEMPTS = int(os.getenv("JOB_MAX_ATTEMPTS", "5"))

# Exponential backoff between attempts: base * 2^(attempt-1), capped, plus jitter.
RETRY_BASE_SECONDS = float(os.getenv("RETRY_BASE_SECONDS", "5"))
RETRY_MAX_SECONDS = float(os.getenv("RETRY_MAX_SECONDS", "300"))

# Global pause after a 429 when the provider doesn't say how long to wait.
RATE_LIMIT_COOLDOWN_SECONDS = float(os.getenv("RATE_LIMIT_COOLDOWN_SECONDS", "60"))

# Matching is interactive: fewer retries, and a bounded wait for a rate-limit slot.
MATCH_MAX_ATTEMPTS = int(os.getenv("MATCH_MAX_ATTEMPTS", "3"))
MATCH_WAIT_TIMEOUT_SECONDS = float(os.getenv("MATCH_WAIT_TIMEOUT_SECONDS", "45"))
# How many times the model may return an invalid matching JSON before we give up.
MATCH_PARSE_ATTEMPTS = int(os.getenv("MATCH_PARSE_ATTEMPTS", "2"))

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
