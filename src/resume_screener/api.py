"""Thin FastAPI wrapper around the pipeline.

POST /screen   run the pipeline (body optional) and return the batch summary
GET  /results  return the full results of the most recent run
GET  /health
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import Settings
from .github_enrichment import GitHubEnricher
from .llm import build_llm_client
from .pipeline import run_pipeline
from .report import to_json_dict, write_csv, write_html, write_json

app = FastAPI(title="Resume Screener", version="1.0.0")

DEFAULT_INPUT = Path("resumes")
DEFAULT_OUTPUT = Path("output/results.json")
_lock = asyncio.Lock()
_latest: dict | None = None


class ScreenRequest(BaseModel):
    input_dir: str = str(DEFAULT_INPUT)
    use_llm: bool = True
    use_github: bool = True


def _safe_input_dir(raw: str) -> Path:
    """Only allow directories inside the working directory (no arbitrary file reads)."""
    path = Path(raw).resolve()
    root = Path.cwd().resolve()
    if root != path and root not in path.parents:
        raise HTTPException(400, "input_dir must be inside the project directory")
    if not path.is_dir():
        raise HTTPException(404, f"input_dir not found: {raw}")
    return path


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/screen")
async def screen(req: ScreenRequest | None = None) -> dict:
    global _latest
    req = req or ScreenRequest()
    input_dir = _safe_input_dir(req.input_dir)
    if _lock.locked():
        raise HTTPException(409, "A screening run is already in progress")
    async with _lock:
        settings = Settings.from_env()
        llm = build_llm_client(settings) if req.use_llm else None
        github = GitHubEnricher(settings) if req.use_github else None
        try:
            result = await run_pipeline(input_dir, settings, llm=llm, github=github)
        finally:
            if github is not None:
                await github.aclose()
        write_json(result, DEFAULT_OUTPUT)
        write_csv(result, DEFAULT_OUTPUT.with_suffix(".csv"))
        write_html(result, DEFAULT_OUTPUT.with_suffix(".html"))
        _latest = to_json_dict(result)
    return {"summary": _latest["summary"], "results_url": "/results"}


@app.get("/results")
async def results() -> dict:
    if _latest is not None:
        return _latest
    if DEFAULT_OUTPUT.is_file():
        return json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    raise HTTPException(404, "No results yet. POST /screen first.")
