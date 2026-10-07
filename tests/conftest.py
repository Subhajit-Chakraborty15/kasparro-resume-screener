import pytest

from resume_screener.config import Settings
from resume_screener.extraction import parse_resume
from resume_screener.ingestion import normalize_text
from resume_screener.models import RawDocument


def make_resume(text: str, name: str = "x.txt", links: list[str] | None = None):
    return parse_resume(RawDocument(path=name, text=normalize_text(text), links=links or [], file_hash=name))


@pytest.fixture
def settings() -> Settings:
    return Settings(github_cache_path=None)


AGENTIC = """Asha Rao
asha@example.com | github.com/asha-dev
SKILLS
Python, FastAPI, Redis, PostgreSQL, Docker, GCP
PROJECTS
Support Agent | LangGraph, FastAPI
• Built a stateful multi-agent workflow with LangGraph, RAG retrieval over embeddings in pgvector and tool calling.
• Evaluation harness measured precision@5 improving from 0.61 to 0.83; async FastAPI with Redis caching and retries.
EXPERIENCE
Backend Intern, Acme
• Developed Python services with FastAPI and PostgreSQL.
"""

WRAPPER = """Priya Nair
priya@example.com
SKILLS
Python, Streamlit, OpenAI API
PROJECTS
AI Chatbot
• Made a chatbot using OpenAI API and Streamlit.
"""

JAVA_ONLY = """Neha Iyer
neha@example.com
SKILLS
Java, Spring Boot, React, JavaScript, MySQL
PROJECTS
Hospital System | Java, Spring Boot, React
• Built a full stack system with Spring Boot REST APIs and a React frontend.
"""

PYTHON_NO_AI = """Vikram Singh
vikram@example.com
SKILLS
Python, Django, PostgreSQL, Redis, Docker
PROJECTS
Order API | Django
• Designed REST APIs with Django and PostgreSQL deployed with Docker.
"""

AI_NO_PYTHON = """Li Wei
liwei@example.com
SKILLS
JavaScript, TypeScript, React, LangChain.js
PROJECTS
Chat UI | TypeScript, LangChain
• Built a RAG assistant in TypeScript with embeddings and a vector database.
"""

JS_PLUS_PYTHON_AI = """Karan Shah
karan@example.com
SKILLS
JavaScript, React, Next.js, Python, FastAPI
PROJECTS
Research Copilot | Next.js, FastAPI, LangChain
• Built RAG over papers using embeddings in pgvector with tool calling; Next.js frontend and FastAPI backend.
"""

SKILLS_ONLY_AI = """Sneha Pillai
sneha@example.com
SKILLS
Python, Flask, LangChain, RAG, LLM
PROJECTS
Library System | Flask, SQLite
• Developed a library management web app with Flask and SQLite supporting issue and return workflows.
"""
