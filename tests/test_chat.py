from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

settings.GEMINI_API_KEY = None

client = TestClient(app)

def test_chat_endpoint_empty_messages():
    """Empty messages should return 400 Bad Request."""
    payload = {
        "messages": [],
        "resume_skills": [],
        "missing_skills": [],
        "job_recommendations": []
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 400


def test_chat_endpoint_fallback_skills():
    """Chat endpoint should return rule-based skill recommendations in fallback mode."""
    payload = {
        "messages": [
            {"role": "user", "content": "what skills should I learn or improve?"}
        ],
        "resume_skills": ["Python", "SQL"],
        "missing_skills": ["Docker", "FastAPI", "React"],
        "job_recommendations": []
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    text = data["response"]
    assert "Skill Development Roadmap" in text
    assert "Docker" in text
    assert "FastAPI" in text


def test_chat_endpoint_fallback_jobs():
    """Chat endpoint should return rule-based job tips in fallback mode."""
    payload = {
        "messages": [
            {"role": "user", "content": "which recommended jobs are best?"}
        ],
        "resume_skills": ["Python"],
        "missing_skills": [],
        "job_recommendations": [
            {"title": "Backend Engineer", "company": "Stripe", "match_score": 85, "missing_skills": []}
        ]
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    text = data["response"]
    assert "Personalized Job Search Advice" in text
    assert "Backend Engineer" in text
    assert "Stripe" in text


def test_chat_endpoint_fallback_role_suggestions():
    """Chat endpoint should suggest role directions from parsed resume skills."""
    payload = {
        "messages": [
            {"role": "user", "content": "what type of jobs can I find with this resume?"}
        ],
        "resume_text": "Built APIs with Python, FastAPI, PostgreSQL, React, and JavaScript.",
        "resume_skills": ["Python", "FastAPI", "PostgreSQL", "React", "JavaScript"],
        "missing_skills": [],
        "job_recommendations": [],
        "career_goal": "backend engineer"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    text = data["response"]
    assert "Suggested Job Directions" in text
    assert "Backend Engineer" in text
