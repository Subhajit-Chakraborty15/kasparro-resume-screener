import pytest

from resume_screener.eligibility import evaluate_eligibility

from conftest import (AGENTIC, AI_NO_PYTHON, JAVA_ONLY, JS_PLUS_PYTHON_AI, PYTHON_NO_AI, SKILLS_ONLY_AI, WRAPPER, make_resume)


def test_agentic_python_candidate_is_eligible():
    r = evaluate_eligibility(make_resume(AGENTIC))
    assert r.eligible and not r.rejection_reasons
    assert r.ai_strength == "strong"
    assert r.python_evidence and r.ai_evidence


def test_java_react_only_profile_is_rejected_with_both_reasons():
    r = evaluate_eligibility(make_resume(JAVA_ONLY))
    assert not r.eligible
    assert "No evidence of Python stack" in r.rejection_reasons
    assert any("AI" in x for x in r.rejection_reasons)
    assert {"Java", "React", "Spring Boot"} <= set(r.matched_skills)


def test_python_without_ai_is_rejected():
    r = evaluate_eligibility(make_resume(PYTHON_NO_AI))
    assert not r.eligible
    assert r.rejection_reasons == ["No AI/agentic project evidence"]


def test_ai_without_python_is_rejected():
    r = evaluate_eligibility(make_resume(AI_NO_PYTHON))
    assert not r.eligible
    assert r.rejection_reasons == ["No evidence of Python stack"]


def test_javascript_alongside_python_and_ai_is_not_rejected():
    r = evaluate_eligibility(make_resume(JS_PLUS_PYTHON_AI))
    assert r.eligible
    assert "JavaScript" in r.matched_skills and "React" in r.matched_skills


def test_thin_wrapper_is_still_eligible_but_skills_only_ai_too():
    assert evaluate_eligibility(make_resume(WRAPPER)).eligible
    assert evaluate_eligibility(make_resume(SKILLS_ONLY_AI)).eligible


def test_python_implied_by_python_only_framework():
    text = "Ravi K\nSKILLS\nFastAPI, Docker\nPROJECTS\nAgent Service\n• Built a LangGraph multi-agent system with tool calling on FastAPI."
    r = evaluate_eligibility(make_resume(text))
    assert r.eligible
    assert r.python_evidence[0].startswith("Implied Python stack")


def test_classical_ml_only_is_rejected_with_specific_reason():
    text = "Meera K\nSKILLS\nPython, Pandas, scikit-learn\nPROJECTS\nTitanic model\n• Trained a random forest on the Titanic dataset with scikit-learn."
    r = evaluate_eligibility(make_resume(text))
    assert not r.eligible
    assert "classical ML" in r.rejection_reasons[0]


def test_deep_learning_counts_only_when_used_in_a_project():
    used = "Aman S\nSKILLS\nPython\nPROJECTS\nImage Classifier\n• Trained a PyTorch convolutional neural network for plant disease detection."
    listed = "Aman S\nSKILLS\nPython, PyTorch\nPROJECTS\nTodo App\n• Built a todo web app with Flask and SQLite for managing daily tasks."
    assert evaluate_eligibility(make_resume(used)).eligible
    assert not evaluate_eligibility(make_resume(listed)).eligible


def test_python_substring_in_other_word_does_not_count():
    text = "Joe B\nSKILLS\nJava, Spring\nPROJECTS\nPythonista Club Site\n• Built a LangChain RAG demo in Java."
    assert "No evidence of Python stack" in evaluate_eligibility(make_resume(text)).rejection_reasons


@pytest.mark.parametrize("text", ["", "   \n  ", "Just a name"])
def test_empty_or_tiny_text_never_crashes(text):
    r = evaluate_eligibility(make_resume(text))
    assert not r.eligible
