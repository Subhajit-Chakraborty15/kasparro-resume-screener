from resume_screener.models import GitHubSummary, LLMAssessment, LLMProjectAssessment
from resume_screener.scoring import score_candidate

from conftest import AGENTIC, JS_PLUS_PYTHON_AI, SKILLS_ONLY_AI, WRAPPER, make_resume


def score(text, settings, gh=None, llm=None):
    return score_candidate(make_resume(text), gh, settings, llm)


def test_breakdown_respects_category_caps(settings):
    s = score(AGENTIC, settings, gh=GitHubSummary(status="ok", score=10, summary="x"))
    b = s.breakdown
    assert b.ai_project_depth <= 40 and b.python_backend <= 30 and b.cloud_fullstack <= 15
    assert b.github <= 10 and b.engineering_depth <= 5
    assert s.total == sum(b.model_dump().values()) - sum(p.points for p in s.penalties)


def test_agentic_project_outranks_thin_wrapper(settings):
    assert score(AGENTIC, settings).total > score(WRAPPER, settings).total + 30


def test_thin_wrapper_is_penalised_within_5_to_15(settings):
    s = score(WRAPPER, settings)
    thin = [p for p in s.penalties if "thin" in p.reason]
    assert thin and 5 <= thin[0].points <= 15


def test_framework_in_skills_only_earns_little_ai_credit(settings):
    s = score(SKILLS_ONLY_AI, settings)
    assert s.breakdown.ai_project_depth <= settings.skills_only_ai_cap
    assert any("no project-level AI evidence" in c for c in s.concerns)


def test_weak_ai_candidate_with_strong_backend_is_capped(settings):
    text = """Strong Backend
SKILLS
Python, FastAPI, asyncio, PostgreSQL, Redis, Docker, GCP, pytest, LangChain
PROJECTS
Orders Platform | FastAPI, PostgreSQL, Redis
• Designed async FastAPI services with PostgreSQL, Redis caching, Celery queues and retries; deployed on GCP Cloud Run with Docker and CI/CD.
• Added pytest unit tests, structured logging and monitoring.
EXPERIENCE
Backend Intern, Acme
• Built Python REST APIs with FastAPI handling 2k requests per minute.
"""
    s = score(text, settings, gh=GitHubSummary(status="ok", score=10, summary="very active"))
    assert s.breakdown.ai_project_depth < settings.weak_ai_threshold
    assert s.raw_total > settings.weak_ai_total_cap  # would have ranked high without the guard
    assert s.total == settings.weak_ai_total_cap and s.capped


def test_skill_used_in_project_beats_skill_only_in_list(settings):
    used = "A B\nSKILLS\nPython, LangChain\nPROJECTS\nSvc | FastAPI\n• Built a RAG agent with LangChain and Redis caching on FastAPI.\n"
    listed = "A B\nSKILLS\nPython, LangChain, Redis, FastAPI\nPROJECTS\nTodo\n• Built a todo list web app with Flask for personal task tracking and reminders.\n"
    assert score(used, settings).breakdown.python_backend > score(listed, settings).breakdown.python_backend


def test_fullstack_credit_requires_frontend_and_backend_together(settings):
    s = score(JS_PLUS_PYTHON_AI, settings)
    assert any("3.0/3" in e and "Full stack" in e for e in s.evidence["cloud_fullstack"])


def test_github_points_come_from_summary_and_missing_github_is_zero(settings):
    assert score(AGENTIC, settings, gh=GitHubSummary(status="ok", score=7, summary="x")).breakdown.github == 7
    assert score(AGENTIC, settings, gh=None).breakdown.github == 0
    assert score(AGENTIC, settings, gh=GitHubSummary(status="rate_limited")).breakdown.github == 0


def test_llm_hybrid_blends_scores_when_evidence_is_present(settings):
    det = score(AGENTIC, settings).breakdown.ai_project_depth
    llm = LLMAssessment(ai_project_depth=20, projects=[LLMProjectAssessment(name="p", is_ai_project=True, evidence=["Built a stateful multi-agent workflow"])])
    s = score(AGENTIC, settings, llm=llm)
    assert s.scoring_mode == "hybrid"
    assert s.breakdown.ai_project_depth == round(0.5 * 20 + 0.5 * det)


def test_llm_score_without_verifiable_evidence_is_ignored(settings):
    det = score(AGENTIC, settings)
    llm = LLMAssessment(ai_project_depth=40, projects=[LLMProjectAssessment(name="p", is_ai_project=True, evidence=[])])
    s = score(AGENTIC, settings, llm=llm)
    assert s.scoring_mode == "deterministic"
    assert s.breakdown.ai_project_depth == det.breakdown.ai_project_depth


def test_total_never_negative(settings):
    s = score(WRAPPER, settings)
    assert s.total >= 0
