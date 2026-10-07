"""Generate SYNTHETIC resumes (fictional people) for demos and tests.

    python scripts/make_sample_resumes.py sample_resumes
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

RESUMES: dict[str, str] = {}

RESUMES["asha_rao"] = """Asha Rao
asha.rao@example.com | github.com/asha-rao-dev | Bengaluru
SUMMARY
Backend engineer focused on agentic systems.
SKILLS
Python, FastAPI, asyncio, PostgreSQL, Redis, Docker, GCP, LangGraph, LangChain, pytest
PROJECTS
Support Triage Agent | LangGraph, FastAPI, PostgreSQL, Redis
• Built a stateful multi-agent workflow with LangGraph that routes tickets, retrieves answers via RAG and calls tools (function calling) to update the CRM.
• Implemented retrieval with embeddings in pgvector, hybrid search and reranking; evaluation harness measured precision@5 improving from 0.61 to 0.83.
• Async FastAPI backend with Redis caching, retries with exponential backoff and structured logging; deployed on GCP Cloud Run with Docker and CI/CD.
• Wrote pytest unit and integration tests covering 85% of the agent graph.
Document QA Service | Python, LlamaIndex, FAISS
• Designed a RAG pipeline that ingests 5,000 PDFs, chunking and embeddings stored in FAISS, with an evaluation set to track hallucination rate.
EXPERIENCE
Backend Intern, Nimbus Labs
• Developed Python microservices with FastAPI and PostgreSQL handling 2k requests per minute.
EDUCATION
B.Tech Computer Science, Sample University
"""

RESUMES["rohan_mehta"] = """Rohan Mehta
rohan.mehta@example.com | https://github.com/rohan-mehta-ml
SKILLS
Python, Flask, SQL, Docker, LangChain, OpenAI
PROJECTS
PDF Chat Assistant | Python, LangChain, Chroma
• Built a RAG application that chunks PDFs, creates embeddings stored in ChromaDB and answers questions with OpenAI models.
• Added a simple evaluation script comparing answers with a small test set.
Expense Tracker | Flask, SQLite
• Developed a Flask REST API with SQLite for tracking expenses.
EDUCATION
B.E. Information Technology, Sample Institute
"""

RESUMES["priya_nair_wrapper"] = """Priya Nair
priya.nair@example.com | github.com/priya-nair-codes
SKILLS
Python, Streamlit, OpenAI API, HTML, CSS
PROJECTS
AI Chatbot
• Made a chatbot using OpenAI API and Streamlit.
Portfolio Website
• Built a personal portfolio website with HTML and CSS.
EDUCATION
B.Tech Computer Science
"""

RESUMES["vikram_singh_backend_only"] = """Vikram Singh
vikram.singh@example.com | github.com/vikram-singh-be
SKILLS
Python, Django, FastAPI, PostgreSQL, Redis, Celery, Docker, AWS, pytest
PROJECTS
Order Management API | Django, PostgreSQL, Redis
• Designed REST APIs with Django and PostgreSQL, Redis caching, Celery background jobs; deployed on AWS with Docker and CI/CD.
• Added pytest unit tests and structured logging.
EXPERIENCE
Software Engineering Intern, Cartwheel
• Built Python services with FastAPI and async workers.
EDUCATION
B.Tech Computer Science
"""

RESUMES["neha_iyer_java"] = """Neha Iyer
neha.iyer@example.com | github.com/neha-iyer-java
SKILLS
Java, Spring Boot, MySQL, React, JavaScript, Docker, AWS
PROJECTS
Hospital Management System | Java, Spring Boot, React
• Built a full stack hospital management system with Spring Boot REST APIs and a React frontend, deployed on AWS.
EDUCATION
B.Tech Information Technology
"""

RESUMES["arjun_das_mern"] = """Arjun Das
arjun.das@example.com
SKILLS
JavaScript, TypeScript, React, Next.js, Node.js, MongoDB, Docker
PROJECTS
E-commerce Platform | Next.js, Node.js, MongoDB
• Built a full stack e-commerce platform with Next.js and Node.js, deployed on Vercel.
EDUCATION
B.Tech Computer Science
"""

RESUMES["meera_krishnan_ds"] = """Meera Krishnan
meera.krishnan@example.com | github.com/meera-k-data
SKILLS
Python, Pandas, NumPy, scikit-learn, SQL, Tableau
PROJECTS
Titanic Survival Prediction | Python, scikit-learn
• Built a random forest regression model on the Titanic dataset with 82% accuracy.
House Price Prediction
• Used linear regression on the Boston housing dataset.
EDUCATION
B.Sc Statistics
"""

RESUMES["karan_shah_js_python_rag"] = """Karan Shah
karan.shah@example.com | github.com/karan-shah-fs
SKILLS
JavaScript, TypeScript, React, Next.js, Node.js, Python, FastAPI, PostgreSQL, Docker, GCP, LangChain
PROJECTS
Research Copilot | Next.js, FastAPI, LangChain, pgvector
• Built an end-to-end research assistant: Next.js frontend, FastAPI backend, RAG over uploaded papers using embeddings in pgvector and tool calling for web search.
• Implemented async ingestion workers and caching; deployed on GCP Cloud Run with Docker and GitHub Actions CI/CD.
• Evaluated retrieval quality with recall@5 on a hand labelled test set (0.78).
EDUCATION
B.Tech Computer Science
"""

RESUMES["sneha_pillai_skills_only"] = """Sneha Pillai
sneha.pillai@example.com | github.com/sneha-pillai-dev
SKILLS
Python, Flask, LangChain, RAG, LLM, SQL, React
PROJECTS
Library Management System | Python, Flask, SQLite
• Developed a library management web app with Flask and SQLite supporting book issue and return workflows.
Weather Dashboard | React
• Created a weather dashboard that calls a public API and renders charts.
EDUCATION
B.Tech Computer Science
"""

RESUMES["ishaan_verma_tutorial"] = """Ishaan Verma
ishaan.verma@example.com | github.com/ishaan-verma-ai
SKILLS
Python, TensorFlow, OpenAI
PROJECTS
LLM Chatbot (Udemy tutorial)
• Followed a Udemy tutorial to build a chatbot with the OpenAI API.
EDUCATION
B.Tech Computer Science
"""

RESUMES["tanvi_joshi_cv_agents"] = """TANVI JOSHI
tanvi.joshi@example.com
TECHNICAL SKILLS
Python, Google ADK, LangGraph, FastAPI, Docker, Redis
PROJECTS
Travel Planner Agent | Google ADK, Python
• Developed a multi-agent travel planner with Google ADK where a planner agent delegates to flight and hotel tool-calling agents and keeps session state.
• Added guardrails and an evaluation set of 40 prompts; reduced tool errors by 35% with retries and fallbacks.
EXPERIENCE
AI Engineering Intern, Orbit AI
• Built Python FastAPI services around LLM agents with async calls and Redis caching.
"""

TXT_RESUME = """Dev Patel
dev.patel@example.com
GitHub: dev-patel-llm

Skills: Python, FastAPI, PostgreSQL, Docker, LangChain

Projects
Contract Analyzer - Python, LangChain, FAISS
- Built a RAG pipeline for contract clauses using embeddings and FAISS, with a FastAPI API and evaluation of retrieval precision.
- Implemented async document ingestion and error handling for malformed PDFs.
"""


def build_pdf(path: Path, text: str, link: tuple[str, str] | None = None) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    story = []
    for line in text.strip().split("\n"):
        if not line.strip():
            story.append(Spacer(1, 6))
            continue
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe, styles["BodyText"]))
    if link:
        label, url = link
        story.append(Paragraph(f'<link href="{url}">{label}</link>', styles["BodyText"]))
    doc.build(story)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, text in RESUMES.items():
        build_pdf(out / f"{name}.pdf", text)
    # hidden-URL GitHub link (anchor text only says "GitHub")
    build_pdf(out / "link_only_github.pdf",
              "Zoya Khan\nzoya.khan@example.com\nSKILLS\nPython, FastAPI, LangGraph, Docker\nPROJECTS\n"
              "Agent Router | LangGraph, FastAPI\n• Built a stateful multi-agent router with tool calling and evaluation metrics using FastAPI and Redis.",
              link=("GitHub", "https://github.com/zoya-khan-agents"))
    shutil.copy(out / "asha_rao.pdf", out / "asha_rao_copy.pdf")  # duplicate file
    (out / "corrupt.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf \x00\x01\x02")
    (out / "empty.pdf").write_bytes(b"")
    (out / "dev_patel.txt").write_text(TXT_RESUME)
    (out / "notes.xlsx").write_bytes(b"PK not a resume")
    try:
        import docx

        d = docx.Document()
        for line in ["Maya Reddy", "maya.reddy@example.com | github.com/maya-reddy-ai", "SKILLS",
                     "Python, FastAPI, PostgreSQL, Docker, LlamaIndex, LangGraph", "PROJECTS",
                     "Clinical Notes Assistant | LlamaIndex, FastAPI",
                     "• Built a RAG service with embeddings and vector search over clinical notes, with tool calling and an evaluation set; async FastAPI backend with PostgreSQL."]:
            d.add_paragraph(line)
        d.save(out / "maya_reddy.docx")
    except ImportError:
        pass
    print(f"Wrote {len(list(out.iterdir()))} files to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "sample_resumes"))
