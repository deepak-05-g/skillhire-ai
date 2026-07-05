import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.matcher import rank_jobs

client = TestClient(app)


SAMPLE_RESUME_TEXT = """
John Doe
Software Engineer with Machine Learning experience.
Projects: Built a REST API with FastAPI and React dashboard.
Skills: Python, React, FastAPI, SQL, Git
"""

PYTHON_HEAVY_JOB = {
    "source": "Greenhouse",
    "title": "Backend Software Engineer",
    "company": "TechCo",
    "location": "Remote",
    "description": "Build backend services with Python and FastAPI.",
    "requirements": "Python, FastAPI, SQL, Git, Docker, AWS",
    "apply_url": "https://example.com/job1",
    "job_type": "Full-time",
}

REACT_HEAVY_JOB = {
    "source": "Lever",
    "title": "Frontend Developer Intern",
    "company": "WebCo",
    "location": "New York",
    "description": "Internship building React UIs.",
    "requirements": "React, JavaScript, TypeScript, CSS",
    "apply_url": "https://example.com/job2",
    "job_type": "Internship",
}


def _mock_semantic(_resume_text: str, job_text: str) -> float:
    """Deterministic semantic scores for stable unit tests."""
    if "Python" in job_text or "FastAPI" in job_text:
        return 0.85
    if "React" in job_text:
        return 0.55
    return 0.40


@patch("app.services.matcher._semantic_similarity", side_effect=_mock_semantic)
def test_job_with_more_matching_skills_ranks_higher(mock_semantic):
    """A job whose requirements overlap more with the resume should rank higher."""
    resume_skills = ["Python", "FastAPI", "SQL", "Git", "React"]

    results = rank_jobs(SAMPLE_RESUME_TEXT, resume_skills, [REACT_HEAVY_JOB, PYTHON_HEAVY_JOB])

    assert len(results) == 2
    assert results[0]["title"] == "Backend Software Engineer"
    assert results[0]["match_score"] >= results[1]["match_score"]
    assert results[0]["skill_match_score"] > results[1]["skill_match_score"]


@patch("app.services.matcher._semantic_similarity", side_effect=_mock_semantic)
def test_missing_skills_detected_correctly(mock_semantic):
    """Missing job skills not present on the resume should be reported."""
    resume_skills = ["Python", "FastAPI", "SQL", "Git"]

    results = rank_jobs(SAMPLE_RESUME_TEXT, resume_skills, [PYTHON_HEAVY_JOB])
    job = results[0]

    assert "Docker" in job["missing_skills"]
    assert "AWS" in job["missing_skills"]
    assert "Python" in job["matched_skills"]
    assert "FastAPI" in job["matched_skills"]


@patch("app.services.matcher._semantic_similarity", side_effect=_mock_semantic)
def test_recommendation_contains_match_score(mock_semantic):
    """Each ranked job should expose match_score and component scores."""
    resume_skills = ["Python", "React"]

    results = rank_jobs(SAMPLE_RESUME_TEXT, resume_skills, [PYTHON_HEAVY_JOB])

    assert len(results) == 1
    job = results[0]
    assert "match_score" in job
    assert isinstance(job["match_score"], int)
    assert 0 <= job["match_score"] <= 100
    assert "skill_match_score" in job
    assert "semantic_similarity_score" in job
    assert "source" in job
    assert "reason" in job
    assert job.get("fit_label") in {"Low Fit", "Medium Fit", "High Fit"}


@patch("app.services.matcher._semantic_similarity", side_effect=_mock_semantic)
def test_recommend_jobs_api_route(mock_semantic):
    """POST /api/v1/recommend/jobs should return ranked recommendations."""
    payload = {
        "resume_text": SAMPLE_RESUME_TEXT,
        "resume_skills": ["Python", "FastAPI", "SQL", "Git", "React"],
        "jobs": [REACT_HEAVY_JOB, PYTHON_HEAVY_JOB],
    }

    response = client.post("/api/v1/recommend/jobs", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    recs = data["recommendations"]
    assert len(recs) == 2
    assert recs[0]["match_score"] >= recs[1]["match_score"]
    assert all("matched_skills" in r for r in recs)
    assert all("missing_skills" in r for r in recs)


@patch("app.services.matcher._semantic_similarity", side_effect=_mock_semantic)
def test_recommend_jobs_api_rejects_empty_jobs(mock_semantic):
    """Empty jobs list should return HTTP 400."""
    payload = {
        "resume_text": SAMPLE_RESUME_TEXT,
        "resume_skills": ["Python"],
        "jobs": [],
    }

    response = client.post("/api/v1/recommend/jobs", json=payload)
    assert response.status_code == 400
