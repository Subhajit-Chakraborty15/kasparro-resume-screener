"""Central configuration.

Everything tunable (weights, thresholds, model names, concurrency, API keys)
lives here so that business logic never hard-codes it. Secrets are read from
environment variables only (optionally via a local, git-ignored .env file).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4.1-mini"}


@dataclass(frozen=True)
class Weights:
    """Top-level 100 point rubric from the assignment."""

    ai_project_depth: int = 40
    python_backend: int = 30
    cloud_fullstack: int = 15
    github: int = 10
    engineering_depth: int = 5

    def as_dict(self) -> dict[str, int]:
        return {
            "ai_project_depth": self.ai_project_depth,
            "python_backend": self.python_backend,
            "cloud_fullstack": self.cloud_fullstack,
            "github": self.github,
            "engineering_depth": self.engineering_depth,
        }


@dataclass(frozen=True)
class AIDepthWeights:
    """How the 40 AI-project points are split for a single project (sum = 40)."""

    presence_strong: float = 8      # LLM / RAG / agent framework is genuinely used
    presence_moderate: float = 3    # deep learning / NLP only (no LLM evidence)
    orchestration: float = 8        # agents, graphs, state, multi-step workflows
    retrieval: float = 8            # RAG, embeddings, vector search, chunking
    tools: float = 6                # tool / function calling, MCP, structured output
    evaluation: float = 5           # evals, metrics, benchmarks, guardrails
    backend_logic: float = 3        # real data processing / backend / product logic
    ownership: float = 2            # quantified outcome or clear ownership verbs


@dataclass(frozen=True)
class PenaltyConfig:
    thin_wrapper_best: int = 10          # best AI project is a thin LLM wrapper
    thin_wrapper_no_detail_extra: int = 5  # ...and it is also described in <N words
    thin_wrapper_other: int = 5          # a secondary AI project is thin
    tutorial_or_no_detail: int = 5       # tutorial-style / no implementation detail
    max_total: int = 20
    no_detail_words: int = 25


@dataclass(frozen=True)
class Settings:
    weights: Weights = field(default_factory=Weights)
    ai_weights: AIDepthWeights = field(default_factory=AIDepthWeights)
    penalties: PenaltyConfig = field(default_factory=PenaltyConfig)

    # Ranking guard: strong Python without a real AI project must not rank high.
    weak_ai_threshold: int = 12
    weak_ai_total_cap: int = 50
    skills_only_ai_cap: int = 8   # max AI points when AI appears only in a skills list

    # Ingestion
    parse_concurrency: int = 8
    max_file_mb: int = 15

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = DEFAULT_MODELS["anthropic"]
    llm_api_key: str | None = field(default=None, repr=False)
    llm_timeout: float = 60.0
    llm_concurrency: int = 4
    llm_blend: float = 0.5
    llm_max_chars: int = 12000

    # GitHub
    github_token: str | None = field(default=None, repr=False)
    github_concurrency: int = 4
    github_timeout: float = 15.0
    github_max_retries: int = 2
    github_cache_path: Path | None = None
    github_cache_ttl_hours: float = 6.0
    github_activity_days: int = 90
    github_maintained_days: int = 365

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        if load_dotenv is not None:
            load_dotenv()

        anthropic_key = os.getenv("ANTHROPIC_API_KEY") or None
        openai_key = os.getenv("OPENAI_API_KEY") or None
        provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        if not provider:
            provider = "openai" if (openai_key and not anthropic_key) else "anthropic"
        api_key = openai_key if provider == "openai" else anthropic_key
        model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"])

        cache = os.getenv("GITHUB_CACHE_PATH", "output/.github_cache.json")
        return cls(
            llm_provider=provider,
            llm_model=model,
            llm_api_key=api_key,
            llm_timeout=_float("LLM_TIMEOUT_SECONDS", 60.0),
            llm_concurrency=_int("LLM_CONCURRENCY", 4),
            llm_blend=min(1.0, max(0.0, _float("LLM_BLEND", 0.5))),
            github_token=os.getenv("GITHUB_TOKEN") or None,
            github_concurrency=_int("GITHUB_CONCURRENCY", 4),
            github_timeout=_float("GITHUB_TIMEOUT_SECONDS", 15.0),
            github_cache_path=Path(cache) if cache else None,
            github_cache_ttl_hours=_float("GITHUB_CACHE_TTL_HOURS", 6.0),
            parse_concurrency=_int("PARSE_CONCURRENCY", 8),
            max_file_mb=_int("MAX_FILE_MB", 15),
            weak_ai_threshold=_int("WEAK_AI_THRESHOLD", 12),
            weak_ai_total_cap=_int("WEAK_AI_TOTAL_CAP", 50),
        )


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
