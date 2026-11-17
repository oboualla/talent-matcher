"""Streamlit interface for AI_TalentMatcher: a thin client of the FastAPI backend (api.py).

Uploads return immediately; the "Processing & candidates" section polls the API every
couple of seconds while jobs are pending, so new profiles appear as soon as they are ready.
"""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
POLL_SECONDS = 2

st.set_page_config(page_title="AI_TalentMatcher", page_icon="🧑‍💼", layout="wide")

state = st.session_state
state.setdefault("upload_results", [])
state.setdefault("answer", None)
state.setdefault("answer_question", "")
state.setdefault("uploader_key", 0)
state.setdefault("polling", False)

STATUS_LABEL = {
    "queued": "⏳ queued",
    "processing": "⚙️ processing",
    "waiting_retry": "🔁 waiting to retry",
    "done": "✅ done",
    "failed": "❌ failed",
    "cancelled": "🚫 cancelled",
}


class ApiError(Exception):
    pass


def api(method: str, path: str, timeout: float = 10, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"The backend API is unreachable at {API_URL} ({type(exc).__name__}).") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(str(detail))
    return resp.json() if resp.content else None


def full_name(profile: dict) -> str:
    return " ".join(p for p in (profile.get("firstName"), profile.get("lastName")) if p) or "(name not found)"


# ---------- sidebar ----------
with st.sidebar:
    st.header("Backend")
    try:
        status = api("GET", "/status")
        rl = status["rate_limit"]
        st.markdown(f"**Model:** `{status['model']}`")
        st.markdown(f"**Endpoint:** `{status['endpoint']}`")
        st.markdown(f"**Workers:** `{status['workers']}` · **Max attempts:** `{status['max_attempts']}`")
        st.markdown(f"**Rate limit:** {rl['used_last_minute']}/{rl['requests_per_minute'] or '∞'} requests in the last minute")
        cooldown = rl["cooldown_seconds"]
        if cooldown > 300:
            st.error(f"Provider quota exhausted for this model: resets in ~{cooldown / 3600:.1f}h. "
                     "Switch LLM_MODEL or the API key in .env.")
        elif cooldown:
            st.warning(f"Provider asked us to slow down: paused for {cooldown:.0f}s")
        st.caption("Change these in `.env`, then restart with `docker compose up --build`.")
    except ApiError as exc:
        st.error(str(exc))
    st.divider()
    if st.button("Clear all candidates and jobs"):
        try:
            api("DELETE", "/candidates")
            state.answer = None
            state.upload_results = []
        except ApiError as exc:
            st.error(str(exc))
        st.rerun()

st.title("AI_TalentMatcher")
st.caption("From resumes to recommendations: upload CVs → profiles are extracted in the background → ask for candidates.")

# ---------- 1. upload ----------
st.header("1. Upload CVs")
files = st.file_uploader(
    "Text-based PDF resumes (one candidate per file). You can add more at any time.",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"uploader_{state.uploader_key}",
)
if st.button("Submit CVs", type="primary", disabled=not files):
    try:
        payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
        state.upload_results = api("POST", "/cvs", files=payload, timeout=60)
        state.uploader_key += 1  # clear the uploader
        state.polling = True
    except ApiError as exc:
        st.error(f"Upload failed: {exc}")
    st.rerun()

for r in state.upload_results:
    text = f"**{r['filename']}**: {r['message']}"
    if r["outcome"] == "rejected":
        st.error(text + "  \nNo profile will be created for this file.")
    elif r["outcome"] == "queued":
        st.success(text)
    else:
        st.info(text)


# ---------- 2. processing & candidates (auto-refreshing) ----------
def render_live_section() -> None:
    try:
        jobs = api("GET", "/jobs")
        candidates = api("GET", "/candidates")
    except ApiError as exc:
        st.error(str(exc))
        return

    active = [j for j in jobs if j["status"] in ("queued", "processing", "waiting_retry")]
    # Stop polling once everything is finished (a full rerun re-creates the fragment without a timer).
    if state.polling and not active:
        state.polling = False
        st.rerun()
    elif active and not state.polling:
        state.polling = True
        st.rerun()

    st.header("2. Processing")
    if not jobs:
        st.info("Nothing submitted yet.")
    else:
        finished = sum(j["status"] in ("done", "failed") for j in jobs)
        st.progress(finished / len(jobs), text=f"{finished}/{len(jobs)} finished"
                    + (f" · refreshing every {POLL_SECONDS}s" if active else ""))
        rows = []
        for j in jobs:
            detail = j["error"] or ""
            if detail and j["status"] in ("queued", "processing", "waiting_retry"):
                detail = f"previous attempt failed: {detail}"
            if j["status"] == "waiting_retry" and j["retry_in_seconds"] is not None:
                detail = f"next try in {j['retry_in_seconds']:.0f}s — {detail}"
            rows.append({"File": j["filename"], "Status": STATUS_LABEL.get(j["status"], j["status"]),
                         "Attempts": j["attempts"], "Details": detail})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        for j in jobs:
            if j["status"] in ("failed", "cancelled"):
                msg_col, btn_col = st.columns([6, 1], vertical_alignment="center")
                with msg_col:
                    st.error(f"**{j['filename']}** — {STATUS_LABEL[j['status']]} "
                             f"after {j['attempts']} attempt(s).  \n{j['error'] or ''}")
                with btn_col:
                    if st.button("🔁 Retry", key=f"retry_{j['id']}", use_container_width=True):
                        try:
                            api("POST", f"/jobs/{j['id']}/retry")
                            state.polling = True
                        except ApiError as exc:
                            st.error(str(exc))
                        st.rerun()

    st.header("3. Candidate profiles")
    if not candidates:
        st.info("No candidate profiles yet." + (" They will appear here as soon as they are extracted." if active else ""))
        return

    st.dataframe(
        pd.DataFrame([
            {
                "Name": full_name(c["profile"]),
                "Email": c["profile"]["email"],
                "Phone": c["profile"]["phone"],
                "Location": c["profile"]["location"],
                "Skills": ", ".join(c["profile"]["skills"]),
                "File": c["filename"],
            }
            for c in candidates
        ]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Expand a candidate to compare the stored JSON with the text extracted from the PDF.")
    for c in candidates:
        label = f"{full_name(c['profile'])} — {c['filename']}" + ("  ⚠️" if c["warnings"] else "")
        with st.expander(label):
            for w in c["warnings"]:
                st.warning(w)
            left, right = st.columns(2)
            with left:
                st.markdown("**Stored JSON profile**")
                st.json(c["profile"])
            with right:
                st.markdown("**Text extracted from the PDF**")
                st.text_area("CV text", c["cv_text"], height=320, disabled=True,
                             key=f"text_{c['id']}", label_visibility="collapsed")
            if st.button("Remove this candidate", key=f"rm_{c['id']}"):
                try:
                    api("DELETE", f"/candidates/{c['id']}")
                except ApiError as exc:
                    st.error(str(exc))
                st.rerun()


# run_every is decided on each full rerun: poll only while jobs are pending.
st.fragment(run_every=POLL_SECONDS if state.polling else None)(render_live_section)()

# ---------- 4. matching ----------
st.header("4. Find candidates for a project")
question = st.text_area(
    "Describe the project or the skills you need",
    placeholder="I want to build a small dashboard in Python using Streamlit and SQL. "
                "Which candidates have relevant skills, and why?",
    height=110,
)
if st.button("Suggest candidates", type="primary"):
    try:
        with st.spinner("Asking the model… (waits for the rate limit and retries temporary errors)"):
            result = api("POST", "/match", json={"description": question}, timeout=300)
        state.answer = result["answer"]
        state.answer_question = question.strip()
    except ApiError as exc:
        state.answer = None
        st.warning(str(exc))

if state.answer:
    st.subheader("Suggestions")
    st.caption(f"Question: {state.answer_question}")
    st.markdown(state.answer)
    st.caption("Generated by an LLM from the extracted profiles only. Verify before acting on it.")
