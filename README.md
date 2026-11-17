# Talent Matcher

From resumes to recommendations: a web app that turns PDF CVs into structured candidate
profiles with an LLM, then suggests the best candidates for a project you describe.

This is my solution to the **AI_TalentMatcher** project (EPINOIA Labs, *Introduction to
generative AI: prompting, structured output, LLM integration*). The full subject is in
[`AI_TalentMatcher.subject.pdf`](AI_TalentMatcher.subject.pdf).

## The subject in short

- **Upload CVs.** Upload one or more text-based PDF CVs. The text is extracted with a PDF
  library. Unreadable or empty PDFs get a clear error and are not stored.
- **Structured profiles.** The LLM turns each CV into a JSON profile with exactly six keys
  (`firstName`, `lastName`, `phone`, `email`, `location`, `skills`). The app checks keys and
  types before storing a profile and never accepts invalid output. Missing data is never invented.
- **Candidate matching.** The user describes a project. All stored profiles (never the raw CV
  text) go to the LLM in one request. It answers with the matching candidates and explains each
  choice using only the skills in their profiles, or says that no one fits.
- **Constraints.** Any stack (Streamlit or Gradio suggested). No orchestration frameworks such as
  LangChain or LlamaIndex. Profiles live in volatile memory only. Starts with
  `docker compose up --build`. No filtering, embeddings or RAG: those are left for the writeup.
- **Writeup.** Discuss prompt design, JSON validation, extraction errors, grounding, what six
  fields lose, and what happens at thousands of profiles.

## My implementation

Python · FastAPI · Streamlit · PyMuPDF · LiteLLM · Pydantic · Docker Compose.

- Uploads return right away. LLM extraction runs in background workers with a shared
  rate limiter, exponential backoff and retries.
- Strict Pydantic validation, with a correction loop when the model returns invalid JSON, and a
  grounding check that flags values that don't appear in the CV.
- Matching returns a validated JSON assessment per candidate, which code then checks against
  the real profiles, so the model can't invent skills or candidates.
- Works with Groq, Google AI Studio, a local Ollama model, or a free GPU on Kaggle.

## Repository layout

```
.
├── AI_TalentMatcher.subject.pdf   # the project subject
├── CVS/                           # sample (fictional) CVs for testing
└── ai_talentmatcher/              # the application
```

## Quick start

```bash
cd ai_talentmatcher
cp .env.example .env   # add your API key and model
docker compose up --build
```

- UI: http://localhost:8501
- API docs: http://localhost:8100/docs

See [`ai_talentmatcher/README.md`](ai_talentmatcher/README.md) for configuration, the
architecture, design decisions and writeup notes.
