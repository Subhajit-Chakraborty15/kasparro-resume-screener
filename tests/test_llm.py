import asyncio
import json

import pytest

from resume_screener.config import Settings
from resume_screener.llm import LLMClient, LLMError, build_llm_client, parse_assessment, verify_evidence
from resume_screener.models import LLMAssessment, LLMProjectAssessment

GOOD = {"ai_project_depth": 31, "projects": [{"name": "A", "is_ai_project": True, "thin_wrapper": False, "depth_score": 31,
        "evidence": ["Built a stateful multi-agent workflow"], "reason": "ok"}], "project_summary": "s", "strengths": ["x"], "concerns": []}


class Scripted(LLMClient):
    def __init__(self, replies, settings=None):
        super().__init__(settings or Settings(llm_timeout=1))
        self.replies, self.calls = list(replies), 0

    async def _complete(self, system, user):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        if callable(r):
            return await r()
        return r


def test_parse_accepts_fenced_json_and_clamps_scores():
    a = parse_assessment("```json\n" + json.dumps({**GOOD, "ai_project_depth": 99}) + "\n```")
    assert a.ai_project_depth == 40


@pytest.mark.parametrize("raw", ["", "no json", "{not json}", '{"projects": "wrong type"}'])
def test_parse_rejects_garbage(raw):
    with pytest.raises(LLMError):
        parse_assessment(raw)


def test_fabricated_evidence_is_dropped():
    a = LLMAssessment(projects=[LLMProjectAssessment(evidence=["Built a stateful multi-agent workflow", "Won a Nobel prize"])])
    out = verify_evidence(a, "I Built a stateful multi-agent workflow with LangGraph")
    assert out.projects[0].evidence == ["Built a stateful multi-agent workflow"]


async def test_assess_retries_once_after_invalid_output():
    c = Scripted(["oops", json.dumps(GOOD)])
    a = await c.assess("Built a stateful multi-agent workflow")
    assert a.ai_project_depth == 31 and c.calls == 2


async def test_assess_raises_after_two_invalid_outputs():
    with pytest.raises(LLMError):
        await Scripted(["bad", "worse"]).assess("text")


async def test_provider_error_becomes_llm_error():
    with pytest.raises(LLMError, match="RuntimeError"):
        await Scripted([RuntimeError("api down")]).assess("text")


async def test_timeout_becomes_llm_error():
    async def slow():
        await asyncio.sleep(5)

    with pytest.raises(LLMError, match="timed out"):
        await Scripted([slow], Settings(llm_timeout=0.05)).assess("text")


def test_no_api_key_means_no_client_and_unknown_provider_is_safe():
    assert build_llm_client(Settings()) is None
    assert build_llm_client(Settings(llm_api_key="k", llm_provider="nonsense")) is None
