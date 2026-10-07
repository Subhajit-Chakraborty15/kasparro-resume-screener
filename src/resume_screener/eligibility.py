"""Hard eligibility filter. Purely rule-based, never touches an LLM.

A candidate is eligible only if BOTH hold:
  1. Python evidence (explicit mention, or a Python-only ecosystem such as FastAPI)
  2. AI/LLM/agentic evidence (strong anywhere, or moderate deep-learning evidence
     that appears in project/experience text rather than a bare skills list)

Other stacks (JavaScript, Java, React...) never cause rejection on their own.
"""
from __future__ import annotations

import re

from .catalog import (
    CLASSICAL_ML_RE,
    IMPLICIT_PYTHON_RE,
    MODERATE_AI_RE,
    SKILLS,
    STRONG_AI_RE,
    find_skills,
)
from .models import EligibilityResult, ParsedResume

_PY = SKILLS["Python"]
MAX_EVIDENCE = 3
SNIPPET_CHARS = 160


def _snippet(line: str) -> str:
    s = re.sub(r"\s+", " ", line).strip(" •-")
    return s if len(s) <= SNIPPET_CHARS else s[: SNIPPET_CHARS - 3] + "..."


def _matching_lines(text: str, pattern: re.Pattern[str], limit: int = MAX_EVIDENCE) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        if pattern.search(line):
            snip = _snippet(line)
            if snip and snip not in out:
                out.append(snip)
            if len(out) >= limit:
                break
    return out


def python_evidence(resume: ParsedResume) -> list[str]:
    explicit = _matching_lines(resume.text, _PY)
    if explicit:
        return explicit
    implicit = _matching_lines(resume.text, IMPLICIT_PYTHON_RE)
    return [f"Implied Python stack: {s}" for s in implicit]


def ai_evidence(resume: ParsedResume) -> tuple[list[str], str]:
    """Return (evidence snippets, strength)."""
    # Prefer lines from project/experience text, then fall back to the whole resume.
    strong = _matching_lines(resume.work_text, STRONG_AI_RE)
    if not strong:
        strong = _matching_lines(resume.text, STRONG_AI_RE)
    if strong:
        return strong, "strong"
    moderate = _matching_lines(resume.work_text, MODERATE_AI_RE)
    if moderate:
        return moderate, "moderate"
    return [], "none"


def evaluate_eligibility(resume: ParsedResume) -> EligibilityResult:
    matched = find_skills(resume.text)
    reasons: list[str] = []

    py_ev = python_evidence(resume)
    if not py_ev:
        reasons.append("No evidence of Python stack")

    ai_ev, strength = ai_evidence(resume)
    if not ai_ev:
        if CLASSICAL_ML_RE.search(resume.text):
            reasons.append("Only classical ML/data-science evidence; no AI/LLM/agentic project evidence")
        else:
            reasons.append("No AI/agentic project evidence")

    return EligibilityResult(
        eligible=not reasons,
        rejection_reasons=reasons,
        matched_skills=matched,
        python_evidence=py_ev,
        ai_evidence=ai_ev,
        ai_strength=strength,
    )
