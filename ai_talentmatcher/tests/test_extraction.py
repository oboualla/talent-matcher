"""Tests for extraction, validation, retries and matching, with the LLM mocked out."""

import json

import pymupdf
import pytest

from talentmatcher import matching, profiles
from talentmatcher.matching import MatchingInputError, match_candidates
from talentmatcher.pdf_text import PdfTextError, extract_text
from talentmatcher.profiles import CandidateProfile, ExtractionError, extract_profile, parse_profile

CV_TEXT = """Alex Morgan
Email: alex.morgan@example.com | Phone: +212 6 12 34 56 78 | Rabat, Morocco
SKILLS: Python, SQL, Streamlit, Docker
"""

GOOD = {
    "firstName": "Alex",
    "lastName": "Morgan",
    "phone": "+212 6 12 34 56 78",
    "email": "alex.morgan@example.com",
    "location": "Rabat, Morocco",
    "skills": ["Python", "SQL", "Streamlit", "Docker"],
}


def make_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((50, 72), text, fontsize=11)
    return doc.tobytes()


class FakeChat:
    """Replaces llm.chat: returns queued answers and records the messages it received."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


# ---------- PDF ----------

def test_extract_text_from_text_pdf():
    assert "alex.morgan@example.com" in extract_text(make_pdf(CV_TEXT))


def test_empty_pdf_is_rejected():
    with pytest.raises(PdfTextError, match="No extractable text"):
        extract_text(make_pdf(""))


def test_non_pdf_is_rejected():
    with pytest.raises(PdfTextError):
        extract_text(b"this is not a pdf")


# ---------- schema validation ----------

def test_parse_valid_profile():
    assert parse_profile(json.dumps(GOOD)).email == "alex.morgan@example.com"


def test_parse_tolerates_code_fences():
    assert parse_profile("```json\n" + json.dumps(GOOD) + "\n```").firstName == "Alex"


def test_empty_strings_become_null_and_skills_are_deduplicated():
    p = parse_profile(json.dumps({**GOOD, "phone": "  ", "skills": ["Python", "python ", ""]}))
    assert p.phone is None
    assert p.skills == ["Python"]


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        json.dumps([GOOD]),                                   # array instead of object
        json.dumps({k: v for k, v in GOOD.items() if k != "email"}),  # missing key
        json.dumps({**GOOD, "age": "31"}),                    # extra key
        json.dumps({**GOOD, "skills": "Python, SQL"}),         # wrong type
        json.dumps({**GOOD, "phone": 212612345678}),           # number instead of string
        json.dumps({**GOOD, "skills": ["Python", 3]}),         # non-string skill
    ],
)
def test_invalid_outputs_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_profile(bad)


# ---------- extraction loop ----------

def test_extract_retries_with_correction(monkeypatch):
    fake = FakeChat("Sure! Here is the profile:", json.dumps(GOOD))
    monkeypatch.setattr(profiles, "chat", fake)

    profile, warnings = extract_profile(CV_TEXT)

    assert profile.lastName == "Morgan"
    assert len(fake.calls) == 2
    # The second request contains the bad answer and an explanation of the error.
    assert fake.calls[1][-2] == {"role": "assistant", "content": "Sure! Here is the profile:"}
    assert "not valid JSON" in fake.calls[1][-1]["content"]
    assert any("2 attempts" in w for w in warnings)


def test_extract_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(profiles, "chat", FakeChat("bad", "bad", "bad"))
    with pytest.raises(ExtractionError):
        extract_profile(CV_TEXT)


def test_invented_contact_details_are_removed(monkeypatch):
    invented = {**GOOD, "email": "someone.else@example.com", "phone": "+1 555 0100"}
    monkeypatch.setattr(profiles, "chat", FakeChat(json.dumps(invented)))

    profile, warnings = extract_profile(CV_TEXT)

    assert profile.email is None and profile.phone is None
    assert len([w for w in warnings if "removed" in w]) == 2


# ---------- matching ----------

def test_matching_sends_all_profiles_without_cv_text_or_contacts(monkeypatch):
    fake = FakeChat(json.dumps({"required_skills": ["Python"], "constraints": [], "assessments": [{"id": "C1", "location_check": "no constraint", "meets_constraints": True, "matching_skills": ["Python"], "missing_skills": []}], "no_match_reason": None}))
    monkeypatch.setattr(matching, "chat", fake)
    people = [CandidateProfile(**GOOD), CandidateProfile(**{**GOOD, "firstName": "Sam", "skills": ["Java"]})]

    answer = match_candidates("Python dashboard", people)

    prompt = fake.calls[0][-1]["content"]
    assert "**Alex Morgan (C1)**" in answer
    assert '"id": "C1"' in prompt and '"id": "C2"' in prompt
    assert "alex.morgan@example.com" not in prompt  # contact details not sent
    assert "SKILLS:" not in prompt                  # raw CV text not sent


@pytest.mark.parametrize("desc, people", [("Python", []), ("   ", [CandidateProfile(**GOOD)])])
def test_matching_rejects_empty_inputs(desc, people):
    with pytest.raises(MatchingInputError):
        match_candidates(desc, people)
