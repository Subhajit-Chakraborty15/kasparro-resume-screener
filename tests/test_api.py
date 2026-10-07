import pytest
from fastapi.testclient import TestClient

from conftest import AGENTIC, JAVA_ONLY


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / "resumes").mkdir()
    (tmp_path / "resumes" / "a.txt").write_text(AGENTIC)
    (tmp_path / "resumes" / "b.txt").write_text(JAVA_ONLY)
    import resume_screener.api as api
    monkeypatch.setattr(api, "_latest", None)
    return TestClient(api.app)


def test_results_before_screen_is_404(client):
    assert client.get("/results").status_code == 404


def test_screen_then_results(client):
    r = client.post("/screen", json={"use_llm": False, "use_github": False})
    assert r.status_code == 200 and r.json()["summary"]["eligible"] == 1
    res = client.get("/results").json()
    assert res["ranked_candidates"][0]["rank"] == 1 and res["rejected_candidates"][0]["eligible"] is False


def test_input_dir_outside_project_is_rejected(client):
    assert client.post("/screen", json={"input_dir": "/etc", "use_llm": False, "use_github": False}).status_code == 400


def test_missing_dir_is_404(client):
    assert client.post("/screen", json={"input_dir": "nope", "use_llm": False, "use_github": False}).status_code == 404
