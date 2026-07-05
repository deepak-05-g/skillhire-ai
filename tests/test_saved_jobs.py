import os
import sys

from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.main import app
from app.database import SessionLocal
from app.models import SavedJob

client = TestClient(app)

SAMPLE_SAVE_PAYLOAD = {
    "title": "Backend Engineer",
    "company": "TechCo",
    "location": "Remote",
    "apply_url": "https://example.com/jobs/saved-test-1",
    "match_score": 82,
    "missing_skills": ["Docker", "AWS"],
}


def test_save_job_endpoint():
    """POST /jobs/save should persist a bookmarked job."""
    response = client.post("/api/v1/jobs/save", json=SAMPLE_SAVE_PAYLOAD)
    assert response.status_code == 201
    data = response.json()
    assert data["saved_job"]["title"] == "Backend Engineer"
    assert data["saved_job"]["match_score"] == 82
    assert "Docker" in data["saved_job"]["missing_skills"]


def test_list_saved_jobs_endpoint():
    """GET /jobs/saved should return saved bookmarks."""
    client.post("/api/v1/jobs/save", json=SAMPLE_SAVE_PAYLOAD)
    response = client.get("/api/v1/jobs/saved")
    assert response.status_code == 200
    jobs = response.json()["saved_jobs"]
    assert len(jobs) >= 1
    assert jobs[0]["company"] == "TechCo"


def test_save_job_duplicate_returns_existing():
    """Saving the same apply_url twice should not create duplicates."""
    client.post("/api/v1/jobs/save", json=SAMPLE_SAVE_PAYLOAD)
    response = client.post("/api/v1/jobs/save", json=SAMPLE_SAVE_PAYLOAD)
    assert response.status_code == 201
    assert "already saved" in response.json()["message"].lower()

    db = SessionLocal()
    try:
        count = db.query(SavedJob).filter(
            SavedJob.apply_url == SAMPLE_SAVE_PAYLOAD["apply_url"]
        ).count()
        assert count == 1
    finally:
        db.close()


def test_delete_saved_job_endpoint():
    """DELETE /jobs/saved/{id} should remove a bookmark."""
    save_resp = client.post("/api/v1/jobs/save", json={
        **SAMPLE_SAVE_PAYLOAD,
        "apply_url": "https://example.com/jobs/saved-test-delete",
    })
    saved_id = save_resp.json()["saved_job"]["id"]

    delete_resp = client.delete(f"/api/v1/jobs/saved/{saved_id}")
    assert delete_resp.status_code == 200

    db = SessionLocal()
    try:
        assert db.query(SavedJob).filter(SavedJob.id == saved_id).first() is None
    finally:
        db.close()


def test_delete_saved_job_not_found():
    """Deleting a missing id should return 404."""
    response = client.delete("/api/v1/jobs/saved/999999")
    assert response.status_code == 404
