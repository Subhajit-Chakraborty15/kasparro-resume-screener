"""Deterministic, explainable scoring (100 points) with optional LLM blending.

Design rule: a skill only earns full credit when it is *used* in a project or
role (``work_text``). A skill that only appears in a skills list earns 40%.
Every category records the evidence behind its points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import (
    AI_FEATURES,
    BACKEND_SIGNAL_RE,
    BACKEND_TERMS,
    DEPLOY_RE,
    ENGINEERING_SIGNALS,
    FRONTEND_RE,
    MODERATE_AI_RE,
    OWNERSHIP_RE,
    QUANTIFIED_RE,
    SKILLS,
    STRONG_AI_RE,
    STRONG_AI_SKILLS,
    TUTORIAL_RE,
    IMPLICIT_PYTHON_RE,
    find_skills,
)
from .config import Settings
from .models import (
    GitHubSummary,
    LLMAssessment,
    ParsedResume,
    Penalty,
    ProjectBlock,
    ScoreBreakdown,
    ScoreResult,
)

SKILLS_ONLY_CREDIT = 0.4  # listed in a skills section but not shown in use


# ------------------------------------------------------------------ helpers
def _credit(pattern, resume: ParsedResume) -> float:
    if pattern.search(resume.work_text):
        return 1.0
    if pattern.search(resume.text):
        return SKILLS_ONLY_CREDIT
    return 0.0


def _where(credit: float) -> str:
    return "used in projects/work" if credit >= 1.0 else "skills list only"


# ------------------------------------------------------- AI project analysis
@dataclass
class ChunkAnalysis:
    title: str
    section: str
    words: int
    level: str                      # "strong" | "moderate"
    features: set[str] = field(default_factory=set)
    backend_hits: int = 0
    points: float = 0.0
    thin: bool = False
    tutorial: bool = False
    skills: list[str] = field(default_factory=list)
    first_line: str = ""


def analyze_chunk(block: ProjectBlock, settings: Settings) -> ChunkAnalysis | None:
    """Score one project/role block, or None if it is not an AI project."""
    text = block.text
    if STRONG_AI_RE.search(text):
        level = "strong"
    elif MODERATE_AI_RE.search(text):
        level = "moderate"
    else:
        return None

    w = settings.ai_weights
    feats = {name for name, pat in AI_FEATURES.items() if pat.search(text)}
    backend_hits = sum(1 for pat in BACKEND_TERMS.values() if pat.search(text))

    pts = w.presence_strong if level == "strong" else w.presence_moderate
    pts += w.orchestration if "orchestration" in feats else 0
    pts += w.retrieval if "retrieval" in feats else 0
    pts += w.tools if "tools" in feats else 0
    pts += w.evaluation if "evaluation" in feats else 0
    pts += w.backend_logic if backend_hits >= 2 else (w.backend_logic / 3 if backend_hits == 1 else 0)
    if QUANTIFIED_RE.search(text):
        pts += w.ownership
    elif OWNERSHIP_RE.search(text):
        pts += w.ownership / 2

    has_depth = bool(feats & {"orchestration", "retrieval", "tools", "evaluation"})
    words = len(text.split())
    bullets = [ln.strip(" •-–—*") for ln in text.split("\n")[1:] if ln.strip()]
    return ChunkAnalysis(
        title=block.title,
        section=block.section,
        words=words,
        level=level,
        features=feats,
        backend_hits=backend_hits,
        points=min(pts, 40.0),
        thin=(not has_depth) and backend_hits <= 1,
        tutorial=bool(TUTORIAL_RE.search(text)),
        skills=[s for s in find_skills(text) if s in STRONG_AI_SKILLS][:6],
        first_line=(bullets[0] if bullets else text.split("\n")[0])[:160],
    )


def score_ai_depth(resume: ParsedResume, settings: Settings) -> tuple[float, list[ChunkAnalysis], list[str]]:
    chunks = [c for c in (analyze_chunk(b, settings) for b in resume.projects) if c]
    chunks.sort(key=lambda c: c.points, reverse=True)
    evidence: list[str] = []
    if chunks:
        best = chunks[0]
        second = chunks[1].points if len(chunks) > 1 else 0.0
        score = min(40.0, best.points + 0.25 * second)
        evidence.append(
            f"Best AI project '{best.title}': {sorted(best.features) or ['no depth features']} "
            f"-> {best.points:.0f}/40"
        )
        if len(chunks) > 1:
            evidence.append(f"Second AI project '{chunks[1].title}' adds 25% of {chunks[1].points:.0f}")
        return score, chunks, evidence

    n_strong = sum(1 for name in STRONG_AI_SKILLS if SKILLS[name].search(resume.text))
    score = min(float(settings.skills_only_ai_cap), 2.0 * n_strong)
    evidence.append(f"AI terms appear only outside project/experience text ({n_strong} terms) -> capped at {score:.0f}")
    return score, chunks, evidence


def compute_penalties(chunks: list[ChunkAnalysis], settings: Settings) -> list[Penalty]:
    cfg = settings.penalties
    out: list[Penalty] = []
    if not chunks:
        return out
    best = chunks[0]
    if best.thin:
        pts = cfg.thin_wrapper_best
        if best.words < cfg.no_detail_words:
            pts += cfg.thin_wrapper_no_detail_extra
        out.append(Penalty(
            reason=f"Best AI project '{best.title}' looks like a thin LLM/API wrapper "
                   f"(no orchestration, retrieval, tool use, evaluation or real backend logic)",
            points=pts,
        ))
    elif any(c.thin for c in chunks[1:]):
        out.append(Penalty(reason="A secondary AI project looks like a thin LLM/API wrapper", points=cfg.thin_wrapper_other))
    no_detail = best.words < cfg.no_detail_words
    if best.tutorial:
        out.append(Penalty(reason=f"Best AI project '{best.title}' is tutorial-style", points=cfg.tutorial_or_no_detail))
    elif no_detail and not best.thin:  # thin wrappers already carry the no-detail surcharge
        out.append(Penalty(
            reason=f"Best AI project '{best.title}' is described in fewer than {cfg.no_detail_words} words (no implementation detail)",
            points=cfg.tutorial_or_no_detail,
        ))
    total = sum(p.points for p in out)
    if total > cfg.max_total:  # scale down proportionally to respect the cap
        scale = cfg.max_total / total
        out = [Penalty(reason=p.reason, points=max(1, round(p.points * scale))) for p in out]
    return out


# ------------------------------------------------------ Python and backend
def score_python_backend(resume: ParsedResume, settings: Settings) -> tuple[float, list[str]]:
    """30 points: Python 6, framework 6, async 4, SQL/PostgreSQL 4, Redis 4, API design 3, work evidence 3."""
    ev: list[str] = []
    total = 0.0

    py = _credit(SKILLS["Python"], resume)
    if py == 0:
        py = 0.8 * _credit(IMPLICIT_PYTHON_RE, resume)
    total += 6 * py
    ev.append(f"Python: {6 * py:.1f}/6 ({_where(py) if py else 'not found'})")

    fa, fl, dj = (_credit(SKILLS[n], resume) for n in ("FastAPI", "Flask", "Django"))
    fw = max(fa, 0.67 * max(fl, dj))
    total += 6 * fw
    ev.append(f"Web framework: {6 * fw:.1f}/6 (FastAPI={fa:.1f}, Flask/Django={max(fl, dj):.1f})")

    asy = _credit(SKILLS["Async"], resume)
    total += 4 * asy
    ev.append(f"Async programming: {4 * asy:.1f}/4")

    pg = _credit(SKILLS["PostgreSQL"], resume)
    other_db = max(_credit(SKILLS[n], resume) for n in ("MySQL", "MongoDB", "SQL"))
    db = max(pg, 0.5 * other_db)
    total += 4 * db
    ev.append(f"PostgreSQL/SQL: {4 * db:.1f}/4 (PostgreSQL={pg:.1f}, other DB={other_db:.1f})")

    rd = _credit(SKILLS["Redis"], resume)
    total += 4 * rd
    ev.append(f"Redis: {4 * rd:.1f}/4")

    api = 1.0 if BACKEND_TERMS["api"].search(resume.work_text) or BACKEND_TERMS["service"].search(resume.work_text) else 0.0
    total += 3 * api
    ev.append(f"API/service design in projects: {3 * api:.1f}/3")

    exp = resume.sections.get("experience", "")
    work_py = 1.0 if exp and (SKILLS["Python"].search(exp) or IMPLICIT_PYTHON_RE.search(exp)
                              or BACKEND_TERMS["api"].search(exp)) else 0.0
    total += 3 * work_py
    ev.append(f"Python/backend in internship or job: {3 * work_py:.1f}/3")
    return min(30.0, total), ev


# ----------------------------------------------------- cloud / full stack
def score_cloud_fullstack(resume: ParsedResume, settings: Settings) -> tuple[float, list[str]]:
    """15 points: cloud provider 5 (GCP) or 4 (AWS/Azure), Docker 4, deployment 3, end-to-end frontend 3."""
    ev: list[str] = []
    gcp = _credit(SKILLS["GCP"], resume)
    other_cloud = max(_credit(SKILLS["AWS"], resume), _credit(SKILLS["Azure"], resume))
    cloud = max(5 * gcp, 4 * other_cloud)
    ev.append(f"Cloud provider: {cloud:.1f}/5 (GCP={gcp:.1f}, AWS/Azure={other_cloud:.1f})")

    dock = 4 * _credit(SKILLS["Docker"], resume)
    ev.append(f"Docker: {dock:.1f}/4")

    if DEPLOY_RE.search(resume.work_text):
        deploy = 3.0
    elif DEPLOY_RE.search(resume.text):
        deploy = 3 * SKILLS_ONLY_CREDIT
    else:
        deploy = 0.0
    ev.append(f"Deployment/CI-CD: {deploy:.1f}/3")

    full = 0.0
    if any(FRONTEND_RE.search(b.text) and BACKEND_SIGNAL_RE.search(b.text) for b in resume.projects):
        full = 3.0
        ev.append("Full stack: 3.0/3 (frontend + backend in the same project)")
    elif FRONTEND_RE.search(resume.work_text):
        full = 1.5
        ev.append("Full stack: 1.5/3 (frontend used, no backend alongside)")
    elif FRONTEND_RE.search(resume.text):
        full = 3 * SKILLS_ONLY_CREDIT
        ev.append(f"Full stack: {full:.1f}/3 (skills list only)")
    else:
        ev.append("Full stack: 0.0/3")
    return min(15.0, cloud + dock + deploy + full), ev


# -------------------------------------------------------- engineering depth
def score_engineering_depth(resume: ParsedResume, settings: Settings) -> tuple[float, list[str], list[str]]:
    hits: list[str] = []
    total = 0.0
    for name, pat in ENGINEERING_SIGNALS.items():
        credit = _credit(pat, resume)
        if credit:
            total += credit
            hits.append(name if credit >= 1 else f"{name} (list only)")
    cap = settings.weights.engineering_depth
    return min(float(cap), total), [f"Signals: {', '.join(hits) or 'none'}"], hits


# ------------------------------------------------------------------- GitHub
def github_points(gh: GitHubSummary | None, settings: Settings) -> int:
    return min(settings.weights.github, gh.score) if gh else 0


# ---------------------------------------------------------------- top level
def score_candidate(
    resume: ParsedResume,
    github: GitHubSummary | None,
    settings: Settings,
    llm: LLMAssessment | None = None,
) -> ScoreResult:
    w = settings.weights
    evidence: dict[str, list[str]] = {}

    det_ai, chunks, ai_ev = score_ai_depth(resume, settings)
    ai = det_ai
    mode = "deterministic"
    notes: list[str] = []
    penalties = compute_penalties(chunks, settings)

    if llm is not None:
        if llm.all_evidence or llm.ai_project_depth <= 10:
            ai = settings.llm_blend * llm.ai_project_depth + (1 - settings.llm_blend) * det_ai
            mode = "hybrid"
            ai_ev.append(
                f"Hybrid: LLM {llm.ai_project_depth}/40 x {settings.llm_blend:.2f} + deterministic {det_ai:.0f}/40"
            )
            llm_ai_projects = [p for p in llm.projects if p.is_ai_project]
            if llm_ai_projects and all(p.thin_wrapper for p in llm_ai_projects) and not any(
                "thin" in p.reason.lower() for p in penalties
            ):
                penalties.append(Penalty(reason="LLM judged every AI project to be a thin wrapper", points=settings.penalties.thin_wrapper_best))
        else:
            notes.append("LLM score ignored: it cited no verifiable evidence")
    evidence["ai_project_depth"] = ai_ev

    py, py_ev = score_python_backend(resume, settings)
    evidence["python_backend"] = py_ev
    cloud, cloud_ev = score_cloud_fullstack(resume, settings)
    evidence["cloud_fullstack"] = cloud_ev
    eng, eng_ev, eng_hits = score_engineering_depth(resume, settings)
    evidence["engineering_depth"] = eng_ev
    gh_pts = github_points(github, settings)
    evidence["github"] = [github.summary if github and github.summary else "No GitHub data"]

    breakdown = ScoreBreakdown(
        ai_project_depth=min(w.ai_project_depth, round(ai)),
        python_backend=min(w.python_backend, round(py)),
        cloud_fullstack=min(w.cloud_fullstack, round(cloud)),
        github=gh_pts,
        engineering_depth=min(w.engineering_depth, round(eng)),
    )
    raw_total = sum(breakdown.model_dump().values())
    penalty_total = min(sum(p.points for p in penalties), settings.penalties.max_total)
    total = max(0, raw_total - penalty_total)

    capped = False
    if breakdown.ai_project_depth < settings.weak_ai_threshold and total > settings.weak_ai_total_cap:
        total = settings.weak_ai_total_cap
        capped = True

    strengths, concerns = _strengths_and_concerns(resume, chunks, breakdown, github, eng_hits, penalties, capped, settings)
    if llm is not None and mode == "hybrid":
        strengths = _merge(llm.strengths[:3], strengths)   # semantic LLM points first, rule-based fill the rest
        concerns = _merge(llm.concerns[:3], concerns)
    summary = _project_summary(chunks, llm if mode == "hybrid" else None)

    result = ScoreResult(
        breakdown=breakdown, penalties=penalties, raw_total=raw_total, total=total, capped=capped,
        scoring_mode=mode, evidence=evidence, project_summary=summary, strengths=strengths[:5], concerns=concerns[:6],
    )
    result.evidence["notes"] = notes
    return result


# ---------------------------------------------------------------- narrative
def _merge(base: list[str], extra: list[str]) -> list[str]:
    seen = {s.lower() for s in base}
    return base + [s for s in extra if s and s.lower() not in seen]


def _project_summary(chunks: list[ChunkAnalysis], llm: LLMAssessment | None) -> str:
    if llm and llm.project_summary.strip():
        return llm.project_summary.strip()
    if not chunks:
        return "No project-level AI work found; AI terms appear only in skills or summary text."
    best = chunks[0]
    feats = ", ".join(sorted(best.features)) or "no orchestration/retrieval/tool/eval depth"
    stack = ", ".join(best.skills) or "AI stack not named"
    return f"{best.title}: {stack}; depth signals: {feats}. Evidence: \"{best.first_line}\""


def _strengths_and_concerns(resume, chunks, bd, github, eng_hits, penalties, capped, settings):
    strengths: list[str] = []
    concerns: list[str] = []
    if chunks:
        best = chunks[0]
        labels = {
            "orchestration": "agentic/orchestration workflow",
            "retrieval": "retrieval/RAG pipeline",
            "tools": "tool calling / structured outputs",
            "evaluation": "evaluation of model output",
        }
        for f in ("orchestration", "retrieval", "tools", "evaluation"):
            if f in best.features:
                strengths.append(f"{labels[f][0].upper() + labels[f][1:]} in '{best.title}'")
    work = resume.work_text
    if SKILLS["FastAPI"].search(work):
        strengths.append("FastAPI backend used in projects/work")
    if SKILLS["Async"].search(work):
        strengths.append("Async programming in practice")
    if SKILLS["Docker"].search(work) and (SKILLS["GCP"].search(work) or SKILLS["AWS"].search(work) or SKILLS["Azure"].search(work)):
        strengths.append("Containerised and cloud deployed")
    if github and github.status in ("ok", "partial") and github.score >= 6:
        strengths.append("Strong recent GitHub activity")

    if not chunks:
        concerns.append("AI skills listed but no project-level AI evidence")
    for p in penalties:
        concerns.append(p.reason)
    if not SKILLS["Redis"].search(resume.text):
        concerns.append("No Redis evidence")
    if not SKILLS["PostgreSQL"].search(resume.text):
        concerns.append("No PostgreSQL evidence")
    if not SKILLS["Docker"].search(resume.text):
        concerns.append("No Docker evidence")
    if "testing" not in {h.split(" ")[0] for h in eng_hits}:
        concerns.append("No testing evidence")
    if github is None or github.status == "not_provided":
        concerns.append("No GitHub profile found on resume")
    elif github.status not in ("ok", "partial"):
        concerns.append(f"GitHub enrichment unavailable ({github.status})")
    if capped:
        concerns.append(f"Total capped at {settings.weak_ai_total_cap}: no substantive AI project")
    return strengths, concerns
