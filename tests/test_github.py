from datetime import datetime, timedelta, timezone

import httpx
import pytest

from resume_screener.config import Settings
from resume_screener.github_enrichment import GitHubEnricher, build_summary

NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)


def iso(days_ago: int) -> str:
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo(name="r", days=10, fork=False, archived=False, lang="Python", topics=None, desc=""):
    return {"name": name, "pushed_at": iso(days), "fork": fork, "archived": archived,
            "language": lang, "topics": topics or [], "description": desc}


def events(n, days=5):
    return [{"type": "PushEvent", "created_at": iso(days)} for _ in range(n)]


def test_active_maintained_python_profile_scores_high():
    s = build_summary("u", [repo(f"r{i}") for i in range(6)], events(20), NOW, Settings())
    assert s.status == "ok" and s.activity_score == 5 and s.repo_score == 5 and s.score == 10


def test_inactive_profile_scores_zero():
    s = build_summary("u", [repo(days=900)], events(3, days=700), NOW, Settings())
    assert s.score == 0 and s.maintained_repos == 0


def test_forks_and_archived_repos_are_not_maintained():
    s = build_summary("u", [repo(fork=True), repo(archived=True)], [], NOW, Settings())
    assert s.maintained_repos == 0


def test_relevance_uses_language_topics_or_name():
    repos = [repo("notes", lang="Go"), repo("rag-bot", lang="Go"), repo("x", lang="Go", topics=["llm"]), repo("py", lang="Python")]
    s = build_summary("u", repos, [], NOW, Settings())
    assert s.maintained_repos == 4 and s.relevant_repos == 3


def test_score_never_exceeds_ten_and_missing_events_is_partial():
    s = build_summary("u", [repo(f"r{i}") for i in range(30)], None, NOW, Settings())
    assert s.status == "partial" and s.score <= 10


def make_enricher(handler, **kw):
    client = httpx.AsyncClient(base_url="https://api.github.com", transport=httpx.MockTransport(handler))
    return GitHubEnricher(Settings(github_cache_path=None, github_max_retries=kw.get("retries", 1)), client=client, now_fn=lambda: NOW)


def ok_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/repos"):
        return httpx.Response(200, json=[{**repo("rag"), "stargazers_count": 1}])
    return httpx.Response(200, json=events(6))


async def test_success_and_per_run_cache_deduplicates_calls():
    e = make_enricher(ok_handler)
    results = [await e.enrich("Octo"), await e.enrich("octo")]
    assert results[0].status == "ok" and results[0].score > 0
    assert e.network_calls == 2  # repos + events, once per user
    await e.aclose()


async def test_concurrent_requests_for_same_user_share_one_fetch():
    import asyncio
    e = make_enricher(ok_handler)
    await asyncio.gather(*(e.enrich("octo") for _ in range(5)))
    assert e.network_calls == 2


async def test_missing_username_is_not_an_error():
    e = make_enricher(ok_handler)
    s = await e.enrich(None)
    assert s.status == "not_provided" and e.network_calls == 0


async def test_404_is_recorded_not_raised():
    e = make_enricher(lambda r: httpx.Response(404, json={"message": "Not Found"}))
    s = await e.enrich("ghost")
    assert s.status == "not_found" and s.score == 0


async def test_rate_limit_is_recorded_and_short_circuits_later_users():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(403, json={"message": "API rate limit exceeded"}, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1893456000"})

    e = make_enricher(handler)
    first, second = await e.enrich("a"), await e.enrich("b")
    assert first.status == second.status == "rate_limited"
    assert len(calls) == 1  # second user never hit the network


async def test_server_errors_are_retried_then_reported():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(502)

    e = make_enricher(handler, retries=1)
    s = await e.enrich("flaky")
    assert s.status == "error" and len(calls) == 2


async def test_network_failure_never_raises():
    def handler(request):
        raise httpx.ConnectError("boom")

    e = make_enricher(handler, retries=0)
    s = await e.enrich("x")
    assert s.status == "error"


async def test_events_failure_degrades_to_partial_score():
    def handler(request):
        if request.url.path.endswith("/repos"):
            return httpx.Response(200, json=[repo("a"), repo("b"), repo("c")])
        return httpx.Response(500)

    e = make_enricher(handler, retries=0)
    s = await e.enrich("x")
    assert s.status == "partial" and s.repo_score > 0
