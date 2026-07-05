import os
import sys
from unittest.mock import patch

import fitz
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.main import app
from app.database import SessionLocal
from app.models import Job, Recommendation, Resume

client = TestClient(app)


def _make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


SAMPLE_RESUME_TEXT = """
Jane Student
Skills: Python, FastAPI, React, SQL, Git
Projects: Built REST API with FastAPI
"""

SAMPLE_JOB = {
    "source": "Greenhouse",
    "title": "Backend Engineer",
    "company": "TechCo",
    "location": "Remote",
    "description": "Python and FastAPI backend role.",
    "requirements": "Python, FastAPI, SQL, Docker",
    "apply_url": "https://example.com/jobs/backend-1",
    "job_type": "Full-time",
}


@patch("app.services.matcher._semantic_similarity", return_value=0.8)
def test_parse_resume_saved_to_database(mock_semantic):
    """Parsed resumes should be persisted in the resumes table."""
    pdf = _make_pdf(SAMPLE_RESUME_TEXT)
    response = client.post(
        "/api/v1/resume/parse",
        files={"file": ("resume.pdf", pdf, "application/pdf")},
    )
    assert response.status_code == 200

    db = SessionLocal()
    try:
        resumes = db.query(Resume).all()
        assert len(resumes) == 1
        assert resumes[0].filename == "resume.pdf"
        assert "Python" in resumes[0].extracted_skills
    finally:
        db.close()


@patch("app.routers.jobs.fetch_jobs")
def test_fetched_jobs_saved_to_database(mock_fetch):
    """Jobs returned from fetch should be stored in the jobs table."""
    mock_fetch.return_value = [SAMPLE_JOB]

    response = client.get(
        "/api/v1/jobs/fetch",
        params={"source": "greenhouse", "company": "techco"},
    )
    assert response.status_code == 200

    db = SessionLocal()
    try:
        jobs = db.query(Job).all()
        assert len(jobs) == 1
        assert jobs[0].title == "Backend Engineer"
        assert jobs[0].apply_url == SAMPLE_JOB["apply_url"]
    finally:
        db.close()


def test_stored_jobs_endpoint_returns_database_jobs():
    """GET /jobs/stored should expose persisted jobs for matching."""
    db = SessionLocal()
    try:
        db.add(Job(**SAMPLE_JOB))
        db.commit()
    finally:
        db.close()

    response = client.get("/api/v1/jobs/stored")
    assert response.status_code == 200

    data = response.json()
    assert data["count"] == 1
    assert data["jobs"][0]["title"] == "Backend Engineer"
    assert data["jobs"][0]["apply_url"] == SAMPLE_JOB["apply_url"]


@patch("app.services.matcher._semantic_similarity", return_value=0.75)
def test_recommendations_saved_to_database(mock_semantic):
    """Recommendation runs should create recommendation rows."""
    payload = {
        "resume_text": SAMPLE_RESUME_TEXT,
        "resume_skills": ["Python", "FastAPI", "SQL", "Git"],
        "jobs": [SAMPLE_JOB],
    }
    response = client.post("/api/v1/recommend/jobs", json=payload)
    assert response.status_code == 200

    db = SessionLocal()
    try:
        recs = db.query(Recommendation).all()
        assert len(recs) == 1
        assert recs[0].match_score >= 0
        assert "Python" in recs[0].matched_skills
    finally:
        db.close()


@patch("app.services.matcher._semantic_similarity", return_value=0.75)
def test_recommend_stored_jobs_endpoint(mock_semantic):
    """POST /recommend/stored-jobs should rank persisted jobs automatically."""
    db = SessionLocal()
    try:
        db.add(Job(**SAMPLE_JOB))
        db.commit()
    finally:
        db.close()

    payload = {
        "resume_text": SAMPLE_RESUME_TEXT,
        "resume_skills": ["Python", "FastAPI", "SQL", "Git"],
        "job_limit": 10,
    }
    response = client.post("/api/v1/recommend/stored-jobs", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["jobs_analyzed"] == 1
    assert data["recommendations"][0]["title"] == "Backend Engineer"

    db = SessionLocal()
    try:
        assert db.query(Recommendation).count() == 1
    finally:
        db.close()


@patch("app.services.matcher._semantic_similarity", return_value=0.75)
def test_recommend_history_endpoint(mock_semantic):
    """GET /recommend/history should return saved recommendation records."""
    payload = {
        "resume_text": SAMPLE_RESUME_TEXT,
        "resume_skills": ["Python", "FastAPI", "SQL", "Git"],
        "jobs": [SAMPLE_JOB],
    }
    client.post("/api/v1/recommend/jobs", json=payload)

    response = client.get("/api/v1/recommend/history")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert len(data["history"]) == 1

    entry = data["history"][0]
    assert entry["match_score"] >= 0
    assert entry["job"]["title"] == "Backend Engineer"
    assert entry["resume"]["filename"] == "api_recommend"
