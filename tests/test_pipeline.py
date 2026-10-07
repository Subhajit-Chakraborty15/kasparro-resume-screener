import json
from pathlib import Path

import pytest

from resume_screener.config import Settings
from resume_screener.llm import LLMClient, LLMError
from resume_screener.models import GitHubSummary
from resume_screener.pipeline import run_pipeline
from resume_screener.report import to_json_dict, write_csv, write_html, write_json

from conftest import AGENTIC, JAVA_ONLY, PYTHON_NO_AI, WRAPPER


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    d = tmp_path / "resumes"
    d.mkdir()
    (d / "asha.txt").write_text(AGENTIC)
    (d / "asha_copy.txt").write_text(AGENTIC)                       # duplicate content
    (d / "priya.txt").write_text(WRAPPER)
    (d / "neha.txt").write_text(JAVA_ONLY)
    (d / "vikram.txt").write_text(PYTHON_NO_AI)
    (d / "broken.pdf").write_bytes(b"%PDF-1.4 \x00 not a pdf")
    (d / "empty.txt").write_text("")
    (d / "sheet.xlsx").write_bytes(b"PK")
    return d


class FakeGitHub:
    def __init__(self):
        self.seen = []

    async def enrich(self, username):
        self.seen.append(username)
        return GitHubSummary(username=username, status="ok", score=8, summary="active")


class ExplodingLLM(LLMClient):
    async def _complete(self, system, user):
        raise RuntimeError("model unavailable")


async def test_batch_survives_bad_files_and_counts_add_up(folder, settings):
    r = await run_pipeline(folder, settings)
    s = r.summary
    assert s.total_files == 8
    assert s.eligible == 2 and s.rejected == 2 and s.duplicates_skipped == 1
    assert s.failed_or_unreadable == 3  # broken.pdf, empty.txt, sheet.xlsx
    assert s.successfully_parsed == s.eligible + s.rejected + s.duplicates_skipped
    assert {f.source_file for f in r.failed_resumes} == {"broken.pdf", "empty.txt", "sheet.xlsx"}


async def test_ranking_order_ranks_and_rejection_reasons(folder, settings):
    r = await run_pipeline(folder, settings)
    names = [c.candidate_name for c in r.ranked_candidates]
    assert names == ["Asha Rao", "Priya Nair"]
    assert [c.rank for c in r.ranked_candidates] == [1, 2]
    rejected = {c.source_file: c.rejection_reasons for c in r.rejected_candidates}
    assert "No evidence of Python stack" in rejected["neha.txt"]
    assert rejected["vikram.txt"] == ["No AI/agentic project evidence"]


async def test_github_only_checked_for_eligible_candidates(folder, settings):
    gh = FakeGitHub()
    r = await run_pipeline(folder, settings, github=gh)
    assert sorted(gh.seen) == ["asha-dev"]  # Priya has no GitHub; rejected profiles are skipped
    top = r.ranked_candidates[0]
    assert top.score_breakdown.github == 8 and top.github_summary == "active"


async def test_llm_failure_falls_back_to_deterministic_with_warning(folder, settings):
    r = await run_pipeline(folder, settings, llm=ExplodingLLM(settings))
    assert r.summary.llm_failures == 2
    top = r.ranked_candidates[0]
    assert top.scoring_mode == "deterministic" and any("LLM assessment failed" in w for w in top.warnings)


async def test_output_files_have_expected_shape(folder, settings, tmp_path):
    r = await run_pipeline(folder, settings)
    out = tmp_path / "out" / "results.json"
    write_json(r, out)
    write_csv(r, out.with_suffix(".csv"))
    write_html(r, out.with_suffix(".html"))
    data = json.loads(out.read_text())
    assert set(data) == {"summary", "ranked_candidates", "rejected_candidates", "failed_resumes", "duplicate_resumes"}
    first = data["ranked_candidates"][0]
    assert {"rank", "candidate_name", "eligible", "total_score", "score_breakdown", "matched_skills",
            "project_summary", "github_summary", "strengths", "concerns"} <= set(first)
    assert set(first["score_breakdown"]) == {"ai_project_depth", "python_backend", "cloud_fullstack", "github", "engineering_depth"}
    assert {"candidate", "eligible", "rejection_reasons", "matched_skills"} <= set(data["rejected_candidates"][0])
    assert out.with_suffix(".csv").read_text().startswith("rank,candidate_name")
    assert "Asha Rao" in out.with_suffix(".html").read_text()


async def test_empty_folder_returns_empty_result(tmp_path, settings):
    d = tmp_path / "none"
    d.mkdir()
    r = await run_pipeline(d, settings)
    assert r.summary.total_files == 0 and r.ranked_candidates == []


class GoodLLM(LLMClient):
    async def _complete(self, system, user):
        return json.dumps({"ai_project_depth": 36, "project_summary": "LLM summary", "strengths": ["LLM strength"],
                           "projects": [{"name": "Support Agent", "is_ai_project": True, "thin_wrapper": False, "depth_score": 36,
                                         "evidence": ["Built a stateful multi-agent workflow with LangGraph"]}]})


async def test_llm_success_produces_hybrid_scores_with_verified_evidence(folder, settings):
    r = await run_pipeline(folder, settings, llm=GoodLLM(settings))
    top = r.ranked_candidates[0]
    assert top.scoring_mode == "hybrid" and top.project_summary == "LLM summary" and "LLM strength" in top.strengths
    assert r.summary.llm_used and r.summary.llm_failures == 0
