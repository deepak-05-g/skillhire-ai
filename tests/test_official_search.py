import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.job_sources.official_search import (
    build_amazon_careers_url,
    build_google_careers_url,
    build_microsoft_careers_url,
    generate_official_search_sources,
    infer_role,
)


def test_infer_role_intern_from_text():
    """Resume mentioning internship should map to intern role."""
    text = "Computer Science student seeking a software engineering internship."
    assert infer_role(text, ["Python", "Java"]) == "Software Engineering Intern"


def test_google_careers_url_uses_official_domain():
    """Google search URL should point to careers.google.com with query params."""
    url = build_google_careers_url("Software Engineer Python", "India")
    assert url.startswith("https://careers.google.com/jobs/results/")
    assert "q=Software" in url or "q=Software+Engineer" in url
    assert "location=India" in url


def test_microsoft_careers_url_uses_official_domain():
    """Microsoft search URL should use the official careers portal."""
    url = build_microsoft_careers_url("Software Engineer", "India")
    assert url.startswith("https://jobs.careers.microsoft.com/global/en/search")
    assert "q=Software" in url
    assert "lc=India" in url


def test_amazon_careers_url_uses_official_domain():
    """Amazon search URL should use amazon.jobs with base_query and loc_query."""
    url = build_amazon_careers_url("Software Engineer", "India")
    assert url.startswith("https://www.amazon.jobs/en/search")
    assert "base_query=Software" in url
    assert "loc_query=India" in url


def test_generate_official_search_sources_shape():
    """Generated sources should include required fields and null match_score."""
    text = "Software Engineer intern in India with Python and Machine Learning skills."
    skills = ["Python", "Machine Learning", "SQL"]

    sources = generate_official_search_sources(text, skills, location="India")
    assert len(sources) == 3

    google = sources[0]
    assert google["company"] == "Google"
    assert google["match_score"] is None
    assert google["apply_url"].startswith("https://careers.google.com/")
    assert "Python" in google["reason"] or "Machine Learning" in google["reason"]
    assert google["location"] == "India"


def test_official_search_api_route():
    """POST /jobs/official-search should return official search sources."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    payload = {
        "resume_text": "Software Engineer with Python skills based in India.",
        "resume_skills": ["Python", "FastAPI"],
        "location": "India",
    }
    response = client.post("/api/v1/jobs/official-search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "official_search_sources" in data
    assert len(data["official_search_sources"]) >= 2
    assert all(s["match_score"] is None for s in data["official_search_sources"])
