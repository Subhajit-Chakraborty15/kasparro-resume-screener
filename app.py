"""Optional FastAPI interface:  uvicorn app:app --reload"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from resume_screener.api import app  # noqa: E402,F401
