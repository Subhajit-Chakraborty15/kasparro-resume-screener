"""CLI entry point.

    python main.py --input ./resumes --output ./output/results.json
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from resume_screener.config import Settings  # noqa: E402
from resume_screener.github_enrichment import GitHubEnricher  # noqa: E402
from resume_screener.llm import build_llm_client  # noqa: E402
from resume_screener.pipeline import run_pipeline  # noqa: E402
from resume_screener.report import format_terminal, write_csv, write_html, write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen and rank resumes for an SDE intern (Python + AI) role.")
    p.add_argument("--input", "-i", type=Path, default=Path("resumes"), help="folder with resumes (PDF/DOCX/TXT)")
    p.add_argument("--output", "-o", type=Path, default=Path("output/results.json"), help="results JSON path")
    p.add_argument("--no-llm", action="store_true", help="deterministic scoring only (no LLM calls)")
    p.add_argument("--no-github", action="store_true", help="skip GitHub enrichment")
    p.add_argument("--no-cache", action="store_true", help="disable the on-disk GitHub cache")
    p.add_argument("--no-extras", action="store_true", help="write only the JSON file (skip CSV and HTML)")
    p.add_argument("--top", type=int, default=10, help="rows to print in the terminal summary")
    p.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.no_cache:
        settings = dataclasses.replace(settings, github_cache_path=None)
    llm = None if args.no_llm else build_llm_client(settings)
    github = None if args.no_github else GitHubEnricher(settings)
    if not args.no_llm and llm is None:
        print("[info] No LLM API key configured: running in deterministic mode.", file=sys.stderr)

    try:
        result = await run_pipeline(args.input, settings, llm=llm, github=github)
    except NotADirectoryError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    finally:
        if github is not None:
            await github.aclose()

    if result.summary.total_files == 0:
        print(f"[warn] No resumes found in {args.input}. Nothing written; add files and rerun.", file=sys.stderr)
        return 1

    write_json(result, args.output)
    if not args.no_extras:
        write_csv(result, args.output.with_suffix(".csv"))
        write_html(result, args.output.with_suffix(".html"))
    print(format_terminal(result, top=args.top))
    print(f"\nResults written to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # noisy on slightly malformed PDFs
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
