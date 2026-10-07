from pathlib import Path

import pytest

from resume_screener.extraction import classify_heading, find_github, guess_name, split_blocks
from resume_screener.ingestion import ResumeReadError, discover_files, normalize_text, read_document

from conftest import AGENTIC, make_resume


def test_corrupt_pdf_raises_resume_read_error(tmp_path: Path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4 garbage \x00\x01")
    with pytest.raises(ResumeReadError):
        read_document(p)


def test_empty_file_and_unknown_type(tmp_path: Path):
    empty = tmp_path / "e.pdf"
    empty.write_bytes(b"")
    with pytest.raises(ResumeReadError, match="empty"):
        read_document(empty)
    odd = tmp_path / "x.rtf"
    odd.write_text("hello world " * 10)
    with pytest.raises(ResumeReadError, match="Unsupported"):
        read_document(odd)


def test_txt_resume_is_read_and_hashed(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text(AGENTIC)
    doc = read_document(p)
    assert "LangGraph" in doc.text and len(doc.file_hash) == 64


def test_discover_files_separates_supported_hidden_and_unsupported(tmp_path: Path):
    for name in ["a.pdf", "b.docx", "c.txt", "d.xlsx", ".DS_Store"]:
        (tmp_path / name).write_bytes(b"x")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "e.pdf").write_bytes(b"x")
    supported, unsupported = discover_files(tmp_path)
    assert {p.name for p in supported} == {"a.pdf", "b.docx", "c.txt", "e.pdf"}
    assert [p.name for p in unsupported] == ["d.xlsx"]


def test_missing_input_dir_raises(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        discover_files(tmp_path / "nope")


def test_normalize_unifies_bullets_and_cid_artifacts():
    assert normalize_text("(cid:127) Built\n\uf0b7 Did\x7f").count("•") == 3


def test_pdf_hidden_github_link_is_extracted(tmp_path: Path):
    from reportlab.platypus import Paragraph, SimpleDocTemplate
    from reportlab.lib.styles import getSampleStyleSheet

    p = tmp_path / "r.pdf"
    st = getSampleStyleSheet()["BodyText"]
    SimpleDocTemplate(str(p)).build([
        Paragraph("Zoya Khan zoya@example.com Python FastAPI LangGraph agents with tool calling built end to end", st),
        Paragraph('<link href="https://github.com/zoya-khan-agents">GitHub</link>', st),
    ])
    doc = read_document(p)
    assert "https://github.com/zoya-khan-agents" in doc.links
    assert find_github(doc.text, doc.links)[0] == "zoya-khan-agents"


def test_github_prefers_profile_link_over_repo_and_ignores_reserved():
    text = "see github.com/features and https://github.com/someorg/project and github.com/real-user"
    assert find_github(text, []) == ("real-user", "https://github.com/real-user")
    assert find_github("github.com/only-repo-owner/repo-name", [])[0] == "only-repo-owner"
    assert find_github("no links here", []) == (None, None)
    assert find_github("GitHub: dev-patel-llm", [])[0] == "dev-patel-llm"


def test_headings_and_sections():
    assert classify_heading("PROJECTS") == "projects"
    assert classify_heading("Technical Skills:") == "skills"
    assert classify_heading("Work Experience") == "experience"
    assert classify_heading("Built a RAG pipeline for contracts") is None
    assert classify_heading("Experience with Python and FastAPI") is None   # body text, not a heading
    assert classify_heading("Projects completed: 5 in total") is None
    assert classify_heading("Projects & Research") == "projects"
    assert classify_heading("SKILLS & TOOLS") == "skills"
    r = make_resume(AGENTIC)
    assert {"projects", "skills", "experience"} <= set(r.sections)


def test_name_email_extraction_and_fallbacks():
    r = make_resume(AGENTIC)
    assert r.name == "Asha Rao" and r.email == "asha@example.com" and r.github_username == "asha-dev"
    assert guess_name(["resume", "jane.doe@example.com"], "jane.doe@example.com", "f.pdf") == "Jane Doe"
    assert guess_name([], None, "/x/john_smith_cv.pdf") == "John Smith Cv"


def test_split_blocks_separates_projects():
    text = "Proj One | Python\n• Built a thing with FastAPI and Redis.\n• Added tests.\nProj Two | Java\n• Built another thing with Spring.\n"
    blocks = split_blocks(text, "projects")
    assert [b.title for b in blocks] == ["Proj One", "Proj Two"]


def test_resume_without_headings_still_yields_a_work_block():
    r = make_resume("Jane Doe\nI built Python RAG pipelines with LangChain and FastAPI for document search at scale.")
    assert r.projects and "LangChain" in r.work_text
