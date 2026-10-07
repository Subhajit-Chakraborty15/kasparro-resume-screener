"""Typed data models shared across the pipeline."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------- ingestion
class RawDocument(BaseModel):
    path: str
    text: str
    links: list[str] = Field(default_factory=list)
    file_hash: str
    page_count: Optional[int] = None


class ProjectBlock(BaseModel):
    title: str
    text: str
    section: str  # "projects" | "experience" | "other"


class ParsedResume(BaseModel):
    source_file: str
    file_hash: str
    text: str
    name: str
    email: Optional[str] = None
    github_username: Optional[str] = None
    github_url: Optional[str] = None
    links: list[str] = Field(default_factory=list)
    sections: dict[str, str] = Field(default_factory=dict)
    projects: list[ProjectBlock] = Field(default_factory=list)

    @property
    def skills_text(self) -> str:
        return self.sections.get("skills", "")

    @property
    def work_text(self) -> str:
        """Text where skills are *used* (projects/experience), not just listed."""
        parts = [self.sections.get(k, "") for k in ("projects", "experience")]
        joined = "\n".join(p for p in parts if p).strip()
        if joined:
            return joined
        # No recognisable project/experience headings: use everything that is
        # not a skills list, education or contact header.
        skip = {"skills", "education", "certifications", "header"}
        return "\n".join(v for k, v in self.sections.items() if k not in skip)


# -------------------------------------------------------------- eligibility
class EligibilityResult(BaseModel):
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    python_evidence: list[str] = Field(default_factory=list)
    ai_evidence: list[str] = Field(default_factory=list)
    ai_strength: Literal["strong", "moderate", "none"] = "none"


# ------------------------------------------------------------------- GitHub
GitHubStatus = Literal["ok", "partial", "not_provided", "not_found", "rate_limited", "error", "skipped"]


class GitHubSummary(BaseModel):
    username: Optional[str] = None
    status: GitHubStatus = "not_provided"
    error: Optional[str] = None
    events_90d: int = 0
    last_activity: Optional[date] = None
    maintained_repos: int = 0
    relevant_repos: int = 0
    total_public_repos: int = 0
    activity_score: int = 0   # 0-5
    repo_score: int = 0       # 0-5
    score: int = 0            # 0-10
    summary: str = ""


# ---------------------------------------------------------------------- LLM
def _clamp(lo: int, hi: int):
    def _v(value):
        try:
            return max(lo, min(hi, int(round(float(value)))))
        except (TypeError, ValueError):
            return lo
    return _v


class LLMProjectAssessment(BaseModel):
    name: str = ""
    is_ai_project: bool = False
    thin_wrapper: bool = False
    depth_score: int = 0
    evidence: list[str] = Field(default_factory=list)
    reason: str = ""

    _clamp_depth = field_validator("depth_score", mode="before")(_clamp(0, 40))


class LLMAssessment(BaseModel):
    ai_project_depth: int = 0
    projects: list[LLMProjectAssessment] = Field(default_factory=list)
    project_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)

    _clamp_total = field_validator("ai_project_depth", mode="before")(_clamp(0, 40))

    @property
    def all_evidence(self) -> list[str]:
        return [e for p in self.projects for e in p.evidence]


# ------------------------------------------------------------------ scoring
class ScoreBreakdown(BaseModel):
    ai_project_depth: int = 0
    python_backend: int = 0
    cloud_fullstack: int = 0
    github: int = 0
    engineering_depth: int = 0


class Penalty(BaseModel):
    reason: str
    points: int


class ScoreResult(BaseModel):
    breakdown: ScoreBreakdown
    penalties: list[Penalty] = Field(default_factory=list)
    raw_total: int = 0
    total: int = 0
    capped: bool = False
    scoring_mode: Literal["deterministic", "hybrid"] = "deterministic"
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    project_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ outputs
class CandidateResult(BaseModel):
    rank: Optional[int] = None
    candidate_name: str
    source_file: str
    email: Optional[str] = None
    github_url: Optional[str] = None
    eligible: bool
    total_score: Optional[int] = None
    score_breakdown: Optional[ScoreBreakdown] = None
    penalties: list[Penalty] = Field(default_factory=list)
    score_notes: list[str] = Field(default_factory=list)
    scoring_mode: Optional[str] = None
    matched_skills: list[str] = Field(default_factory=list)
    project_summary: Optional[str] = None
    github_summary: Optional[str] = None
    github: Optional[GitHubSummary] = None
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class FailedResume(BaseModel):
    source_file: str
    error: str


class DuplicateResume(BaseModel):
    source_file: str
    duplicate_of: str
    reason: str


class BatchSummary(BaseModel):
    input_dir: str
    total_files: int = 0
    successfully_parsed: int = 0
    eligible: int = 0
    rejected: int = 0
    failed_or_unreadable: int = 0
    duplicates_skipped: int = 0
    llm_used: bool = False
    llm_failures: int = 0
    github_checked: int = 0
    github_failures: int = 0
    elapsed_seconds: float = 0.0
    generated_at: str = ""


class ScreeningResult(BaseModel):
    summary: BatchSummary
    ranked_candidates: list[CandidateResult] = Field(default_factory=list)
    rejected_candidates: list[CandidateResult] = Field(default_factory=list)
    failed_resumes: list[FailedResume] = Field(default_factory=list)
    duplicate_resumes: list[DuplicateResume] = Field(default_factory=list)
