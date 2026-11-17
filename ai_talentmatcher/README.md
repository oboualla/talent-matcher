# AI_TalentMatcher

Upload PDF CVs → profiles are extracted by an LLM **in the background** (rate-limited, with
automatic retries) → describe a project → the LLM suggests candidates, justified only by the
skills in their profiles.

Python · FastAPI · Streamlit · PyMuPDF · LiteLLM · Pydantic · Docker Compose. No orchestration framework.

## Run

1. Get a free API key from [Groq](https://console.groq.com/keys) or [Google AI Studio](https://aistudio.google.com/apikey).
2. Create the config file and put your key and model in it:
   ```bash
   cp .env.example .env
   ```
3. Start:
   ```bash
   docker compose up --build
   ```
4. Open the UI at **http://localhost:8501**. The API docs are at **http://localhost:8100/docs**.

Logs (retries, failures): `docker compose logs -f api`

> **Check your model's free-tier limits.** Some Gemini models allow only ~20 requests per
> *day*. Each CV costs at least 1 request (more if the model returns invalid JSON), and each
> match question costs 1. Set `LLM_REQUESTS_PER_MINUTE` at or below the provider's RPM.

### Free option: your own LLM on a Kaggle GPU

Kaggle has no hosted LLM API, but its notebooks come with a free GPU (weekly quota).
[`kaggle/kaggle_llm_server.ipynb`](kaggle/kaggle_llm_server.ipynb) runs an open model there and
exposes it securely:

```
app (LiteLLM, openai/ provider) ─HTTPS─► *.trycloudflare.com ─► auth_proxy.py (token) ─► Ollama (GPU T4)
```

1. On [kaggle.com](https://www.kaggle.com), verify your phone number (needed for GPU and Internet).
2. **Create → New Notebook → File → Import Notebook** and upload `kaggle/kaggle_llm_server.ipynb`.
3. Session options: **Accelerator = GPU T4 x2**, **Internet = On**.
4. Run the cells in order (~5 min the first time). The second-to-last cell prints a `.env` block.
5. Paste it into `.env` and restart: `docker compose up --build`.
6. Keep the Kaggle tab open (the last cell keeps it alive). Stop the session when you're done to save GPU hours.

Things to know:
- **The URL and token change on every Kaggle session**, so repeat steps 4–5 each time.
- If the session stops, jobs go to `waiting_retry` and resume once you update `.env` and restart.
- `qwen2.5:7b` is the default (good JSON, fast on a T4). `qwen2.5:14b` is more accurate but slower.
- There's no per-request quota, so `LLM_REQUESTS_PER_MINUTE=0`. The GPU is the only limit.
- Never make the notebook public or save a version with outputs: the token is printed in it.

**Even simpler:** on a Mac with 16 GB RAM you can run the same model locally with
[Ollama](https://ollama.com) (`ollama pull qwen2.5:7b`). See option D in `.env.example`.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `groq/llama-3.3-70b-versatile` | LiteLLM model string `provider/model` |
| `GROQ_API_KEY` / `GEMINI_API_KEY` | – | Key for the chosen provider |
| `LLM_API_BASE` / `LLM_API_KEY` | – | OpenAI-compatible endpoint and token (Kaggle, Ollama, vLLM…) |
| `LLM_REQUESTS_PER_MINUTE` | `10` | Shared budget for all LLM calls (0 = unlimited) |
| `WORKER_CONCURRENCY` | `2` | Background threads extracting CVs |
| `JOB_MAX_ATTEMPTS` | `5` | Max attempts per CV on temporary errors (429/503/timeout/network); then `failed` with the last error kept and a Retry button |
| `RETRY_BASE_SECONDS` / `RETRY_MAX_SECONDS` | `5` / `300` | Exponential backoff; a provider wait above the max = quota exhausted, fail fast |
| `RATE_LIMIT_COOLDOWN_SECONDS` | `60` | Global pause after a 429 without a retry hint |
| `MATCH_MAX_ATTEMPTS` / `MATCH_WAIT_TIMEOUT_SECONDS` | `3` / `45` | Retries and max wait for the interactive match request |
| `LLM_JSON_MODE` | `true` | Use the provider's JSON mode for extraction |
| `EXTRACTION_TEMPERATURE` / `MATCHING_TEMPERATURE` | `0` / `0.2` | Sampling temperature per step |
| `EXTRACTION_MAX_ATTEMPTS` | `3` | Correction retries when the model returns an invalid profile |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `MAX_CV_CHARS` / `MAX_UPLOAD_MB` | `20000` / `10` | Input limits |
| `API_HOST_PORT` | `8100` | Host port for the API |

### Tests (LLM mocked, no key needed)

```bash
docker compose run --rm --no-deps api sh -c "pip install -q pytest && python -m pytest -q tests"
```

## Architecture

```
 Streamlit UI (:8501)                         FastAPI (:8100 → 8000 in the container)
 ─────────────────────                        ───────────────────────────────────────
 Submit CVs ── POST /cvs ───────────────────► PDF → text (PyMuPDF, instant)
                                              ├─ unreadable → "rejected" right away
            ◄── 202 [queued|cached|rejected] ─┤─ same SHA-256 already done → "cached" (no LLM call)
                                              └─ else JobManager.submit → queue
 every 2s while jobs are pending:
   GET /jobs, GET /candidates ──────────────► worker threads (WORKER_CONCURRENCY)
                                                 take next ready job (min-heap by ready time)
                                                 rate_limiter.acquire()   ← shared RPM budget
                                                 extract_profile → LLM → JSON → Pydantic
                                                 ├─ ok → candidate stored (memory)
                                                 ├─ 429/503/timeout → WAITING_RETRY with backoff
                                                 │    (429 pauses ALL workers for the provider's delay)
                                                 ├─ quota reset > RETRY_MAX_SECONDS → FAILED, clear message
                                                 └─ bad key / invalid JSON ×3 → FAILED
 Suggest ── POST /match ────────────────────► all profiles → LLM (rate-limited, ≤3 short retries)
```

| File | Responsibility |
|---|---|
| `talentmatcher/pdf_text.py` | PDF → text; clear error for unreadable/empty PDFs |
| `kaggle/kaggle_llm_server.ipynb`, `kaggle/auth_proxy.py` | Free self-hosted LLM on a Kaggle GPU, protected by a token |
| `talentmatcher/ratelimit.py` | Thread-safe sliding-window RPM limiter plus a global cooldown (`pause`) |
| `talentmatcher/llm.py` | The only place that calls the LLM; classifies errors as retryable or not; reads the provider's retry delay |
| `talentmatcher/profiles.py` | Extraction prompt, Pydantic schema, parsing, correction loop, grounding check |
| `talentmatcher/jobs.py` | In-memory job queue, worker threads, backoff/retry scheduling, hash cache, candidate store |
| `talentmatcher/matching.py` | Matching prompt; sends all stored profiles in one request; short retries |
| `api.py` | REST endpoints (`/cvs`, `/jobs`, `/jobs/{id}/retry`, `/candidates`, `/match`, `/status`) |
| `app.py` | Streamlit client; auto-refreshes while jobs are pending |

Job states: `queued → processing → done`, or `→ waiting_retry → queued` (temporary error),
or `→ failed` (permanent error or too many attempts; the UI has a **Retry** button).

### Design decisions

- **Respond first, work later.** Uploads return `202 Accepted` in milliseconds. Only the slow LLM step is queued.
- **One rate limit for everything.** The quota belongs to the API key, so background extraction and interactive matching share one limiter. A 429 pauses every worker, not just the job that received it.
- **Retry only what can succeed.** 429/503/5xx/timeouts are retried with exponential backoff plus jitter, using the provider's `retryDelay` when given. A bad key, an unknown model, or repeated invalid JSON fails immediately. A daily quota (reset in hours) fails fast with the reset time instead of retrying every few minutes.
- **A waiting job doesn't block others.** Jobs sit in a heap ordered by their ready time.
- **Cache by content hash.** Re-uploading the same PDF (even renamed) reuses the profile without another LLM call.
- **Volatile memory only.** Everything lives in the API process and is lost on restart, as the subject allows. That's why uvicorn must run as a single process.
- **Missing values are `null`.** Validation is strict: all six keys, no extras, exact types. Invalid output is sent back to the model with the error, at most 3 times.
- **Grounding check.** Email/phone not present in the CV text are removed. Names and skills that don't appear literally are flagged.
- **Matching sees profiles only** (ids `C1…Cn`, no CV text, no contact details).

## Writeup notes (subject chapter IV)

- **Schema-valid ≠ correct.** Example: the prompt splits "AWS (EC2, S3, Lambda)" into "AWS EC2", "AWS S3"… The JSON is valid, but those exact strings aren't in the CV, so the grounding check flags them. Was that a good normalisation or an invention?
- **A real matching failure, and its fix.** Asked for "Laravel/MySQL/Redis, only candidates living in France", `qwen2.5:7b` recommended a candidate from Casablanca. It also listed the same skills as both "matching" and "missing". Causes: the prompt only covered skills, not hard constraints; the free-text answer was never validated; and the small model took the "most skills wins" shortcut. Fix: the model now returns a JSON assessment per candidate (`location_check`, `meets_constraints`, matching and missing skills). Code then validates it against the real profiles: skill lists are recomputed, unknown ids are dropped, a constraint can't fail when none was asked, and an unknown location can't pass one. Constraint failures are shown as "not eligible". Candidates are still never *filtered* in code before the LLM call, as the subject requires. Two lessons: the model copied a concrete example from the prompt ("Casablanca … is not in France") into an unrelated question, so prompt examples must be neutral; and the model's free-text "reason" sentences contradicted its own JSON, so the explanation is now built from verified facts.
- **What six fields lose**: years of experience, seniority, project context, education, recency of skills. A skills list can't tell a senior developer from a beginner.
- **Scaling to thousands of profiles**: every profile is sent on every question. Roughly 60–100 tokens each means 1,000 candidates ≈ 80k tokens per question. That approaches context limits, is slow, burns free-tier quotas (some allow 20 requests/day), and degrades quality.
- **Rate limits in practice**: a free tier can be limited per minute *and* per day. The background queue smooths out per-minute limits but can't create daily quota.
- **Next step (not implemented, per the subject)**: pre-select candidates with a keyword filter, or with embeddings plus vector search (RAG), and send only the top-k profiles to the LLM.

Use only fictional or consented CVs.
