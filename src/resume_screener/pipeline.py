"""End-to-end orchestration.

ingest -> extract -> hard filter -> (LLM + GitHub, bounded concurrency, eligible only)
       -> score -> rank -> result

Failure isolation: each resume is processed inside its own try/except, so one
corrupt file, one LLM error or one GitHub outage never fails the batch.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .eligibility import evaluate_eligibility
from .extraction import parse_resume
from .github_enrichment import GitHubEnricher
from .ingestion import ResumeReadError, discover_files, read_document
from .llm import LLMClient
from .models import (
    BatchSummary,
    CandidateResult,
    DuplicateResume,
    EligibilityResult,
    FailedResume,
    GitHubSummary,
    LLMAssessment,
    ParsedResume,
    ScreeningResult,
)
from .scoring import score_candidate

log = logging.getLogger(__name__)


async def run_pipeline(
    input_dir: Path,
    settings: Settings,
    llm: LLMClient | None = None,
    github: GitHubEnricher | None = None,
) -> ScreeningResult:
    started = time.perf_counter()
    files, unsupported = discover_files(Path(input_dir))
    failed: list[FailedResume] = [
        FailedResume(source_file=str(p.relative_to(input_dir)), error=f"Unsupported file type '{p.suffix or 'none'}'")
        for p in unsupported
    ]

    # ---- 1. ingest + extract (bounded, off the event loop)
    parse_sem = asyncio.Semaphore(max(1, settings.parse_concurrency))

    async def _parse(path: Path) -> ParsedResume | FailedResume:
        async with parse_sem:
            try:
                doc = await asyncio.to_thread(read_document, path, settings.max_file_mb)
                return await asyncio.to_thread(parse_resume, doc)
            except ResumeReadError as exc:
                return FailedResume(source_file=str(path.relative_to(input_dir)), error=str(exc))
            except Exception as exc:
                log.exception("unexpected parse failure for %s", path)
                return FailedResume(source_file=str(path.relative_to(input_dir)), error=f"Parse error: {type(exc).__name__}: {exc}")

    parsed_or_failed = await asyncio.gather(*(_parse(p) for p in files))
    parsed: list[ParsedResume] = []
    for item in parsed_or_failed:
        (parsed if isinstance(item, ParsedResume) else failed).append(item)  # type: ignore[arg-type]

    # ---- 2. duplicates (identical bytes, or same email)
    parsed, duplicates = _dedupe(parsed)

    # ---- 3. hard eligibility (rule-based)
    verdicts: list[tuple[ParsedResume, EligibilityResult]] = [(r, evaluate_eligibility(r)) for r in parsed]
    eligible = [(r, v) for r, v in verdicts if v.eligible]
    rejected = [(r, v) for r, v in verdicts if not v.eligible]

    # ---- 4. enrich eligible candidates (LLM + GitHub), bounded concurrency
    llm_sem = asyncio.Semaphore(max(1, settings.llm_concurrency))
    stats = {"llm_failures": 0, "github_checked": 0, "github_failures": 0}

    async def _llm(resume: ParsedResume) -> tuple[LLMAssessment | None, str | None]:
        if llm is None:
            return None, None
        try:
            async with llm_sem:
                return await llm.assess(resume.text), None
        except Exception as exc:  # isolate per resume (LLMError or any provider error)
            stats["llm_failures"] += 1
            return None, f"LLM assessment failed ({exc}); deterministic scoring used"

    async def _gh(resume: ParsedResume) -> GitHubSummary | None:
        if github is None:
            return None
        if not resume.github_username:
            return GitHubSummary(status="not_provided", summary="No GitHub profile on resume")
        summary = await github.enrich(resume.github_username)
        stats["github_checked"] += 1
        if summary.status not in ("ok", "partial"):
            stats["github_failures"] += 1
        return summary

    async def _enrich(resume: ParsedResume):
        (assessment, llm_warning), gh = await asyncio.gather(_llm(resume), _gh(resume))
        return assessment, llm_warning, gh

    enriched = await asyncio.gather(*(_enrich(r) for r, _ in eligible))

    # ---- 5. score + rank
    ranked: list[CandidateResult] = []
    for (resume, verdict), (assessment, llm_warning, gh) in zip(eligible, enriched):
        try:
            score = score_candidate(resume, gh, settings, assessment)
        except Exception as exc:  # scoring bug on one resume must not kill the batch
            log.exception("scoring failed for %s", resume.source_file)
            failed.append(FailedResume(source_file=resume.source_file, error=f"Scoring error: {type(exc).__name__}: {exc}"))
            continue
        warnings = [llm_warning] if llm_warning else []
        if gh and gh.status not in ("ok", "partial", "not_provided"):
            warnings.append(f"GitHub enrichment {gh.status}: {gh.error or gh.summary}")
        ranked.append(
            CandidateResult(
                candidate_name=resume.name,
                source_file=resume.source_file,
                email=resume.email,
                github_url=resume.github_url,
                eligible=True,
                total_score=score.total,
                score_breakdown=score.breakdown,
                penalties=score.penalties,
                score_notes=_score_notes(score),
                scoring_mode=score.scoring_mode,
                matched_skills=verdict.matched_skills,
                project_summary=score.project_summary,
                github_summary=gh.summary if gh else "GitHub enrichment disabled for this run",
                github=gh,
                strengths=score.strengths,
                concerns=score.concerns,
                evidence={
                    "python": verdict.python_evidence,
                    "ai": verdict.ai_evidence,
                    **score.evidence,
                },
                warnings=warnings,
            )
        )

    ranked.sort(key=lambda c: (-(c.total_score or 0), -(c.score_breakdown.ai_project_depth if c.score_breakdown else 0), c.candidate_name.lower()))
    for i, cand in enumerate(ranked, start=1):
        cand.rank = i

    rejected_out = [
        CandidateResult(
            candidate_name=r.name,
            source_file=r.source_file,
            email=r.email,
            github_url=r.github_url,
            eligible=False,
            matched_skills=v.matched_skills,
            rejection_reasons=v.rejection_reasons,
            evidence={"python": v.python_evidence, "ai": v.ai_evidence},
        )
        for r, v in rejected
    ]
    rejected_out.sort(key=lambda c: c.source_file.lower())

    summary = BatchSummary(
        input_dir=str(input_dir),
        total_files=len(files) + len(unsupported),
        successfully_parsed=len(parsed) + len(duplicates),
        eligible=len(ranked),
        rejected=len(rejected_out),
        failed_or_unreadable=len(failed),
        duplicates_skipped=len(duplicates),
        llm_used=llm is not None,
        llm_failures=stats["llm_failures"],
        github_checked=stats["github_checked"],
        github_failures=stats["github_failures"],
        elapsed_seconds=round(time.perf_counter() - started, 2),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return ScreeningResult(
        summary=summary,
        ranked_candidates=ranked,
        rejected_candidates=rejected_out,
        failed_resumes=sorted(failed, key=lambda f: f.source_file.lower()),
        duplicate_resumes=duplicates,
    )


def _dedupe(resumes: list[ParsedResume]) -> tuple[list[ParsedResume], list[DuplicateResume]]:
    """Keep the first (alphabetical) copy of identical files or same-email resumes."""
    kept: list[ParsedResume] = []
    by_hash: dict[str, str] = {}
    by_email: dict[str, str] = {}
    dups: list[DuplicateResume] = []
    for r in sorted(resumes, key=lambda x: x.source_file.lower()):
        if r.file_hash in by_hash:
            dups.append(DuplicateResume(source_file=r.source_file, duplicate_of=by_hash[r.file_hash], reason="identical file contents"))
            continue
        if r.email and r.email in by_email:
            dups.append(DuplicateResume(source_file=r.source_file, duplicate_of=by_email[r.email], reason=f"same email ({r.email})"))
            continue
        by_hash[r.file_hash] = r.source_file
        if r.email:
            by_email[r.email] = r.source_file
        kept.append(r)
    return kept, dups


def _score_notes(score) -> list[str]:
    notes = list(score.evidence.get("notes", []))
    if score.penalties:
        notes.append(f"Raw {score.raw_total} minus penalties {sum(p.points for p in score.penalties)} = {score.total}" if not score.capped
                     else f"Raw {score.raw_total} minus penalties, then capped to {score.total}")
    elif score.capped:
        notes.append(f"Raw {score.raw_total} capped to {score.total} (no substantive AI project)")
    return notes
