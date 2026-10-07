"""Lightweight GitHub enrichment (max 10 points).

Score = activity (0-5) + repositories (0-5), both explainable:

Activity (0-5) = events tier + recency tier
  events in last 90 days : 0 -> 0, 1-4 -> 1, 5-14 -> 2, 15+ -> 3
  most recent activity   : <=30 days -> 2, <=90 days -> 1, older -> 0

Repositories (0-5) = maintained tier + relevance tier
  maintained (non-fork, non-archived, pushed <=365d): 0 -> 0, 1-2 -> 1, 3-5 -> 2, 6+ -> 3
  relevant (maintained + Python/AI language, topic or name): 0 -> 0, 1-2 -> 1, 3+ -> 2

Failure policy: never raise. 404 / rate limit / network errors become a status on
the returned summary so the batch keeps going. Results are cached per run (and
optionally on disk), and concurrent requests for the same user are de-duplicated.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Settings
from .models import GitHubSummary

log = logging.getLogger(__name__)

API = "https://api.github.com"
EVENT_TYPES = {"PushEvent", "PullRequestEvent", "CreateEvent", "IssuesEvent", "PullRequestReviewEvent", "IssueCommentEvent"}
RELEVANT_LANGS = {"python", "jupyter notebook"}
RELEVANT_RE = re.compile(
    r"\b(?:llm|llms|rag|agents?|agentic|langchain|langgraph|openai|gpt|embeddings?|vector|machine[- ]learning|"
    r"deep[- ]learning|nlp|transformers?|fastapi|python|ai|ml)\b",
    re.IGNORECASE,
)


class GitHubError(Exception):
    def __init__(self, status: str, message: str = ""):
        super().__init__(message or status)
        self.status = status
        self.message = message or status


# ------------------------------------------------------------ pure scoring
def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _tier(value: int, bounds: list[tuple[int, int]]) -> int:
    """bounds: [(min_value, points), ...] ascending; returns the highest tier reached."""
    pts = 0
    for minimum, points in bounds:
        if value >= minimum:
            pts = points
    return pts


def build_summary(
    username: str,
    repos: list[dict[str, Any]],
    events: list[dict[str, Any]] | None,
    now: datetime,
    settings: Settings,
) -> GitHubSummary:
    """Turn raw repo/event data into an explainable 0-10 score."""
    act_cut = now - timedelta(days=settings.github_activity_days)
    maint_cut = now - timedelta(days=settings.github_maintained_days)

    recent_events = [
        e for e in (events or [])
        if e.get("type") in EVENT_TYPES and (ts := _parse_ts(e.get("created_at"))) and ts >= act_cut
    ]
    stamps = [t for t in (_parse_ts(e.get("created_at")) for e in (events or [])) if t]
    stamps += [t for t in (_parse_ts(r.get("pushed_at")) for r in repos if not r.get("fork")) if t]
    last = max(stamps) if stamps else None

    maintained, relevant = 0, 0
    for r in repos:
        pushed = _parse_ts(r.get("pushed_at"))
        if r.get("fork") or r.get("archived") or not pushed or pushed < maint_cut:
            continue
        maintained += 1
        blob = " ".join([r.get("name") or "", r.get("description") or "", " ".join(r.get("topics") or [])])
        if (r.get("language") or "").lower() in RELEVANT_LANGS or RELEVANT_RE.search(blob):
            relevant += 1

    events_tier = _tier(len(recent_events), [(0, 0), (1, 1), (5, 2), (15, 3)])
    age_days = (now - last).days if last else None
    recency_tier = 0 if age_days is None else (2 if age_days <= 30 else 1 if age_days <= 90 else 0)
    activity = min(5, events_tier + recency_tier)
    repo_score = min(5, _tier(maintained, [(0, 0), (1, 1), (3, 2), (6, 3)]) + _tier(relevant, [(0, 0), (1, 1), (3, 2)]))

    if last is None:
        activity_text = "no recent public activity"
    else:
        activity_text = f"{len(recent_events)} public events in last {settings.github_activity_days} days, last activity {age_days}d ago"
    summary = (
        f"{activity_text}; {maintained} maintained repos ({relevant} Python/AI-relevant) "
        f"of {len(repos)} public. Score {activity}+{repo_score}."
    )
    return GitHubSummary(
        username=username,
        status="ok" if events is not None else "partial",
        events_90d=len(recent_events),
        last_activity=last.date() if last else None,
        maintained_repos=maintained,
        relevant_repos=relevant,
        total_public_repos=len(repos),
        activity_score=activity,
        repo_score=repo_score,
        score=activity + repo_score,
        summary=summary if events is not None else summary + " (events unavailable)",
    )


# --------------------------------------------------------------- the client
class GitHubEnricher:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.settings = settings
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "kasparro-resume-screener",
        }
        if settings.github_token:  # token comes from the environment only
            headers["Authorization"] = f"Bearer {settings.github_token}"
        self._client = client or httpx.AsyncClient(
            base_url=API, headers=headers, timeout=settings.github_timeout, follow_redirects=True
        )
        self._owns_client = client is None
        self._sem = asyncio.Semaphore(max(1, settings.github_concurrency))
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._inflight: dict[str, asyncio.Task[GitHubSummary]] = {}
        self._done: dict[str, GitHubSummary] = {}
        self._rate_limited: str | None = None
        self.network_calls = 0
        self._disk = self._load_disk_cache()
        self._disk_dirty = False

    # ----- public API
    async def enrich(self, username: str | None) -> GitHubSummary:
        if not username:
            return GitHubSummary(status="not_provided", summary="No GitHub profile on resume")
        key = username.lower()
        if key in self._done:
            return self._done[key]
        if key not in self._inflight:
            self._inflight[key] = asyncio.ensure_future(self._enrich_uncached(username))
        summary = await self._inflight[key]
        self._done[key] = summary
        return summary

    async def aclose(self) -> None:
        self._save_disk_cache()
        if self._owns_client:
            await self._client.aclose()

    # ----- internals
    async def _enrich_uncached(self, username: str) -> GitHubSummary:
        try:
            cached = self._disk_get(username)
            if cached:
                repos, events = cached
            else:
                repos = await self._get_json(f"/users/{username}/repos", {"per_page": 100, "sort": "pushed", "type": "owner"})
                events = None
                try:
                    events = await self._get_json(f"/users/{username}/events/public", {"per_page": 100})
                except GitHubError as exc:
                    log.info("events unavailable for %s: %s", username, exc.message)
                self._disk_put(username, repos, events)
            return build_summary(username, repos, events, self._now(), self.settings)
        except GitHubError as exc:
            return self._failure(username, exc.status, exc.message)
        except Exception as exc:  # absolute safety net: never fail the batch
            log.exception("unexpected GitHub failure for %s", username)
            return self._failure(username, "error", f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _failure(username: str, status: str, message: str) -> GitHubSummary:
        texts = {
            "not_found": "GitHub profile not found or private",
            "rate_limited": "GitHub API rate limited; enrichment skipped (set GITHUB_TOKEN to raise the limit)",
            "error": f"GitHub enrichment failed: {message}",
        }
        return GitHubSummary(username=username, status=status, error=message, summary=texts.get(status, message))  # type: ignore[arg-type]

    async def _get_json(self, path: str, params: dict[str, Any]) -> Any:
        if self._rate_limited:
            raise GitHubError("rate_limited", self._rate_limited)
        last_error = "unknown error"
        for attempt in range(self.settings.github_max_retries + 1):
            try:
                async with self._sem:
                    self.network_calls += 1
                    resp = await self._client.get(path, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}"
                await asyncio.sleep(0.5 * (3 ** attempt))
                continue
            code = resp.status_code
            if code == 200:
                return resp.json()
            if code == 404:
                raise GitHubError("not_found", "user not found")
            if code in (403, 429):
                body = resp.text.lower()
                if code == 429 or resp.headers.get("x-ratelimit-remaining") == "0" or "rate limit" in body:
                    reset = resp.headers.get("x-ratelimit-reset")
                    when = f" (resets at {datetime.fromtimestamp(int(reset), timezone.utc):%H:%M UTC})" if reset and reset.isdigit() else ""
                    self._rate_limited = f"rate limit exceeded{when}"
                    raise GitHubError("rate_limited", self._rate_limited)
                raise GitHubError("error", f"HTTP {code}")
            if code >= 500:
                last_error = f"HTTP {code}"
                await asyncio.sleep(0.5 * (3 ** attempt))
                continue
            raise GitHubError("error", f"HTTP {code}")
        raise GitHubError("error", last_error)

    # ----- optional disk cache (trimmed payloads only)
    def _load_disk_cache(self) -> dict[str, Any]:
        path = self.settings.github_cache_path
        if not path or not Path(path).is_file():
            return {}
        try:
            data = json.loads(Path(path).read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _disk_get(self, username: str):
        entry = self._disk.get(username.lower())
        ttl = self.settings.github_cache_ttl_hours * 3600
        if entry and time.time() - entry.get("fetched_at", 0) < ttl and entry.get("events") is not None:
            return entry["repos"], entry["events"]
        return None

    def _disk_put(self, username: str, repos: list[dict], events: list[dict] | None) -> None:
        if events is None or not self.settings.github_cache_path:
            return
        self._disk[username.lower()] = {
            "fetched_at": time.time(),
            "repos": [
                {k: r.get(k) for k in ("name", "fork", "archived", "pushed_at", "language", "topics", "description")}
                for r in repos
            ],
            "events": [{"type": e.get("type"), "created_at": e.get("created_at")} for e in events],
        }
        self._disk_dirty = True

    def _save_disk_cache(self) -> None:
        path = self.settings.github_cache_path
        if not path or not self._disk_dirty:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(self._disk))
        except Exception as exc:
            log.warning("could not write GitHub cache: %s", exc)
