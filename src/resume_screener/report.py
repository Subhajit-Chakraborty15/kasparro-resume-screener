"""Output writers: JSON (primary), CSV, a small HTML report and a terminal table."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .models import CandidateResult, ScreeningResult


def to_json_dict(result: ScreeningResult) -> dict:
    """Public JSON shape. Rejected entries follow the assignment's suggested format."""
    ranked = [c.model_dump(mode="json", exclude_none=True) for c in result.ranked_candidates]
    rejected = [
        {
            "candidate": c.candidate_name,
            "source_file": c.source_file,
            "eligible": False,
            "rejection_reasons": c.rejection_reasons,
            "matched_skills": c.matched_skills,
            "evidence_found": c.evidence,
        }
        for c in result.rejected_candidates
    ]
    return {
        "summary": result.summary.model_dump(mode="json"),
        "ranked_candidates": ranked,
        "rejected_candidates": rejected,
        "failed_resumes": [f.model_dump() for f in result.failed_resumes],
        "duplicate_resumes": [d.model_dump() for d in result.duplicate_resumes],
    }


def write_json(result: ScreeningResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_dict(result), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(result: ScreeningResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rank", "candidate_name", "eligible", "total_score", "ai_project_depth", "python_backend",
            "cloud_fullstack", "github", "engineering_depth", "penalties", "matched_skills",
            "github_status", "rejection_reasons", "source_file"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for c in [*result.ranked_candidates, *result.rejected_candidates]:
            bd = c.score_breakdown
            writer.writerow({
                "rank": c.rank or "",
                "candidate_name": c.candidate_name,
                "eligible": c.eligible,
                "total_score": c.total_score if c.total_score is not None else "",
                "ai_project_depth": bd.ai_project_depth if bd else "",
                "python_backend": bd.python_backend if bd else "",
                "cloud_fullstack": bd.cloud_fullstack if bd else "",
                "github": bd.github if bd else "",
                "engineering_depth": bd.engineering_depth if bd else "",
                "penalties": sum(p.points for p in c.penalties) if c.penalties else "",
                "matched_skills": "; ".join(c.matched_skills),
                "github_status": c.github.status if c.github else "",
                "rejection_reasons": "; ".join(c.rejection_reasons),
                "source_file": c.source_file,
            })


def format_terminal(result: ScreeningResult, top: int = 10) -> str:
    s = result.summary
    lines = [
        "=" * 100,
        f"Resumes: {s.total_files} | parsed: {s.successfully_parsed} | eligible: {s.eligible} | rejected: {s.rejected} "
        f"| failed: {s.failed_or_unreadable} | duplicates: {s.duplicates_skipped}",
        f"LLM: {'on' if s.llm_used else 'off'} (failures: {s.llm_failures}) | GitHub checked: {s.github_checked} "
        f"(failures: {s.github_failures}) | {s.elapsed_seconds}s",
        "=" * 100,
        f"{'#':>3}  {'Candidate':<26}{'Total':>6}{'AI':>5}{'Py':>5}{'Cloud':>6}{'GH':>4}{'Eng':>5}{'Pen':>5}  GitHub",
        "-" * 100,
    ]
    for c in result.ranked_candidates[:top]:
        b = c.score_breakdown
        pen = sum(p.points for p in c.penalties)
        gh = c.github.status if c.github else "-"
        lines.append(
            f"{c.rank:>3}  {c.candidate_name[:25]:<26}{c.total_score:>6}{b.ai_project_depth:>5}{b.python_backend:>5}"
            f"{b.cloud_fullstack:>6}{b.github:>4}{b.engineering_depth:>5}{('-' + str(pen)) if pen else '0':>5}  {gh}"
        )
    if len(result.ranked_candidates) > top:
        lines.append(f"... {len(result.ranked_candidates) - top} more in the output file")
    if result.rejected_candidates:
        lines += ["", "Rejected:"]
        for c in result.rejected_candidates[:top]:
            lines.append(f"  - {c.candidate_name} ({c.source_file}): {'; '.join(c.rejection_reasons)}")
        if len(result.rejected_candidates) > top:
            lines.append(f"  ... {len(result.rejected_candidates) - top} more")
    if result.failed_resumes:
        lines += ["", "Failed / unreadable:"]
        lines += [f"  - {f.source_file}: {f.error}" for f in result.failed_resumes]
    return "\n".join(lines)


def write_html(result: ScreeningResult, path: Path) -> None:
    esc = html.escape
    s = result.summary
    rows = []
    for c in result.ranked_candidates:
        b = c.score_breakdown
        pen = sum(p.points for p in c.penalties)
        detail = "".join(f"<li>{esc(x)}</li>" for x in c.strengths) or "<li>-</li>"
        concerns = "".join(f"<li>{esc(x)}</li>" for x in c.concerns) or "<li>-</li>"
        rows.append(
            f"<tr><td>{c.rank}</td><td><b>{esc(c.candidate_name)}</b><br><small>{esc(c.source_file)}</small></td>"
            f"<td class=n>{c.total_score}</td><td class=n>{b.ai_project_depth}/40</td><td class=n>{b.python_backend}/30</td>"
            f"<td class=n>{b.cloud_fullstack}/15</td><td class=n>{b.github}/10</td><td class=n>{b.engineering_depth}/5</td>"
            f"<td class=n>{('-' + str(pen)) if pen else 0}</td>"
            f"<td>{esc(c.project_summary or '')}<br><small>GitHub: {esc(c.github_summary or 'n/a')}</small></td>"
            f"<td><ul>{detail}</ul></td><td><ul>{concerns}</ul></td></tr>"
        )
    rejected = "".join(
        f"<tr><td>{esc(c.candidate_name)}</td><td>{esc(c.source_file)}</td><td>{esc('; '.join(c.rejection_reasons))}</td>"
        f"<td>{esc(', '.join(c.matched_skills))}</td></tr>"
        for c in result.rejected_candidates
    )
    failed = "".join(f"<tr><td>{esc(f.source_file)}</td><td>{esc(f.error)}</td></tr>" for f in result.failed_resumes)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Resume screening report</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}
th,td{{border:1px solid #ddd;padding:6px 8px;vertical-align:top;text-align:left}}th{{background:#f4f4f4}}.n{{text-align:right}}
ul{{margin:0;padding-left:16px}}small{{color:#666}}</style></head><body>
<h1>Resume screening report</h1>
<p>{s.total_files} files | {s.successfully_parsed} parsed | {s.eligible} eligible | {s.rejected} rejected |
{s.failed_or_unreadable} failed | {s.duplicates_skipped} duplicates | LLM {'on' if s.llm_used else 'off'} | {s.generated_at}</p>
<h2>Ranked candidates</h2><table><tr><th>#</th><th>Candidate</th><th>Total</th><th>AI</th><th>Python</th><th>Cloud</th><th>GitHub</th><th>Eng</th><th>Penalty</th><th>Project and GitHub summary</th><th>Strengths</th><th>Concerns</th></tr>{''.join(rows)}</table>
<h2>Rejected</h2><table><tr><th>Candidate</th><th>File</th><th>Reasons</th><th>Matched skills</th></tr>{rejected}</table>
<h2>Failed / unreadable</h2><table><tr><th>File</th><th>Error</th></tr>{failed}</table>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def top_names(result: ScreeningResult, n: int = 3) -> list[CandidateResult]:
    return result.ranked_candidates[:n]
