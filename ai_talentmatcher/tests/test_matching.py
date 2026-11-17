"""Matching: the model's JSON assessment is validated and corrected before being shown.

Scenarios are taken from real answers of qwen2.5:7b that were wrong.
"""

import json

import pytest

from talentmatcher import matching
from talentmatcher.llm import LLMError
from talentmatcher.matching import find_skill, match_candidates
from talentmatcher.profiles import CandidateProfile

ALEX = CandidateProfile(firstName="Alex", lastName="Chen", phone=None, email=None,
                        location="Paris, France", skills=["Python", "Redis", "Express.js", "PostgreSQL"])
OMAR = CandidateProfile(firstName="Omar", lastName="Bouallam", phone=None, email=None,
                        location="Casablanca, Morocco", skills=["Laravel", "MySQL", "Redis"])
NOLOC = CandidateProfile(firstName="Sam", lastName="Doe", phone=None, email=None,
                         location=None, skills=["Laravel"])
PEOPLE = [ALEX, OMAR, NOLOC]  # ids C1, C2, C3

QUESTION = "Laravel project with MySQL and Redis, only candidates living in France"


def answer(assessments, required=("Laravel", "MySQL", "Redis"), constraints=("must live in France",), reason=None):
    return json.dumps({
        "required_skills": list(required),
        "constraints": list(constraints),
        "assessments": assessments,
        "no_match_reason": reason,
    })


def a(id_, meets, matching=(), missing=(), check="..."):
    return {"id": id_, "location_check": check, "meets_constraints": meets,
            "matching_skills": list(matching), "missing_skills": list(missing)}


def run(monkeypatch, *raw_answers):
    queue = list(raw_answers)
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        return queue.pop(0)

    monkeypatch.setattr(matching, "chat", fake_chat)
    return match_candidates(QUESTION, PEOPLE, sleep=lambda s: None), calls


def section(text, title):
    """Return the lines of one '### title' section."""
    if f"### {title}" not in text:
        return ""
    return text.split(f"### {title}", 1)[1].split("###", 1)[0]


# ---------- skill matching ----------

@pytest.mark.parametrize("wanted, skills, expected", [
    ("laravel", ["Laravel"], "Laravel"),
    ("Express", ["Express.js"], "Express.js"),
    ("Laravel", ["Laravel 10"], "Laravel 10"),
    ("SQL", ["MySQL"], None),          # no partial-word match
    ("Java", ["JavaScript"], None),
    ("Redis", ["Python"], None),
])
def test_find_skill(wanted, skills, expected):
    assert find_skill(wanted, skills) == expected


# ---------- the original bug: constraint ignored ----------

def test_candidate_failing_the_constraint_is_never_recommended(monkeypatch):
    raw = answer([
        a("C2", False, ["Laravel", "MySQL", "Redis"], [], "Casablanca, Morocco is not in France"),
        a("C1", True, ["Redis"], ["Laravel", "MySQL"], "Paris, France is in France"),
    ])
    text, _ = run(monkeypatch, raw)

    assert "Alex Chen (C1)" in section(text, "Recommended")
    assert "Omar" not in section(text, "Recommended")
    assert "Omar Bouallam (C2)" in section(text, "Not eligible (constraints not met)")
    assert "Casablanca, Morocco is not in France" in text


def test_unknown_location_cannot_satisfy_a_constraint(monkeypatch):
    raw = answer([a("C3", True, ["Laravel"], [], "probably France")])
    text, _ = run(monkeypatch, raw)
    assert "Sam Doe (C3)" in section(text, "Not eligible (constraints not met)")
    assert "location unknown" in text


def test_invented_constraint_failure_is_overridden_when_no_constraints(monkeypatch):
    # Real failure: no constraint in the question, model still wrote "... is not in France".
    raw = answer([a("C2", False, ["Laravel"], [], "Casablanca, Morocco is not in France")], constraints=())
    text, _ = run(monkeypatch, raw)
    assert "Omar Bouallam (C2)" in section(text, "Recommended")
    assert "not in France" not in text


# ---------- skill lists corrected from the real profile ----------

def test_skill_lists_are_recomputed_and_never_contradict(monkeypatch):
    # Real failure: identical "matching" and "missing" lists; skills the candidate doesn't have.
    raw = answer([a("C1", True, ["Laravel", "MySQL", "Redis"], ["Laravel", "MySQL", "Redis"])])
    text, _ = run(monkeypatch, raw)
    rec = section(text, "Recommended")
    assert "Matching: Redis\n" in rec
    assert "Missing: Laravel, MySQL" in rec
    assert "Has 1 of the 3 requested skills (Redis); ..." in rec


def test_profile_spelling_is_used_for_matching_skills(monkeypatch):
    raw = answer([a("C1", True, ["express"], [])], required=("express",), constraints=())
    text, _ = run(monkeypatch, raw)
    assert "Matching: Express.js" in text


def test_ranked_by_verified_matches(monkeypatch):
    raw = answer([a("C1", True, ["Redis"]), a("C2", True, ["Redis"])], constraints=())
    text, _ = run(monkeypatch, raw)
    rec = section(text, "Recommended")
    assert rec.index("Omar") < rec.index("Alex")  # Omar really has 3/3


# ---------- robustness ----------

def test_unknown_and_duplicate_ids_are_dropped(monkeypatch):
    raw = answer([a("C99", True, ["Redis"]), a("C1", True, ["Redis"]), a("C1", True, ["Redis"])])
    text, _ = run(monkeypatch, raw)
    assert "C99" not in text
    assert text.count("Alex Chen (C1)") == 1


def test_candidates_skipped_by_the_model_are_reported(monkeypatch):
    raw = answer([a("C1", True, ["Redis"])])
    text, _ = run(monkeypatch, raw)
    skipped = section(text, "Not assessed by the model (check manually)")
    assert "Omar Bouallam (C2): Laravel, MySQL, Redis" in skipped
    assert "Sam Doe (C3): Laravel" in skipped


def test_no_suitable_candidate(monkeypatch):
    raw = answer([a("C2", False, ["Laravel"], [], "Casablanca, Morocco is not in France")],
                 reason="Nobody with these skills lives in France.")
    text, _ = run(monkeypatch, raw)
    assert "### No suitable candidate found\nNobody with these skills lives in France." in text


def test_invalid_json_gets_one_correction_then_fails(monkeypatch):
    text, calls = run(monkeypatch, "Here are the candidates: Alex", answer([a("C1", True, ["Redis"])]))
    assert "Alex Chen (C1)" in text
    assert len(calls) == 2 and "not valid JSON" in calls[1][-1]["content"]

    with pytest.raises(LLMError, match="valid matching result"):
        run(monkeypatch, "nope", "still nope")


def test_contact_details_and_cv_text_are_not_sent(monkeypatch):
    profile = CandidateProfile(firstName="A", lastName="B", phone="+33 1 23", email="a@b.c",
                               location="Paris", skills=["Python"])
    calls = []
    monkeypatch.setattr(matching, "chat", lambda m, **kw: calls.append(m) or answer([], constraints=()))
    match_candidates("Python", [profile], sleep=lambda s: None)
    prompt = calls[0][-1]["content"]
    assert "a@b.c" not in prompt and "+33 1 23" not in prompt
