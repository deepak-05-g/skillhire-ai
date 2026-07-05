"""Tests for skill extraction and alias resolution."""

import fitz
from fastapi.testclient import TestClient

from app.main import app
from app.services.skill_extractor import extractor

client = TestClient(app)


def test_basic_skill_extraction():
    """Standard canonical skills should be extracted from text."""
    text = "We are seeking a developer with experience in Python, Docker, and HTML."
    result = extractor.extract(text)

    assert result["skill_count"] == 3
    assert "Python" in result["skills"]
    assert "Docker" in result["skills"]
    assert "HTML" in result["skills"]


def test_case_insensitivity():
    """Skill matching should be case-insensitive."""
    text = "expert in python, DOCKER, and hTmL."
    result = extractor.extract(text)

    assert result["skill_count"] == 3
    assert "Python" in result["skills"]
    assert "Docker" in result["skills"]
    assert "HTML" in result["skills"]


def test_boundary_collisions():
    """Substring collisions like C inside React should be avoided."""
    text1 = "React and JavaScript developer."
    result1 = extractor.extract(text1)

    assert "React" in result1["skills"]
    assert "JavaScript" in result1["skills"]
    assert "C" not in result1["skills"]
    assert "Java" not in result1["skills"]

    text2 = "Proficient in C programming and Java core."
    result2 = extractor.extract(text2)

    assert "C" in result2["skills"]
    assert "Java" in result2["skills"]


def test_alias_resolution():
    """Common aliases should resolve to canonical skill names."""
    text = (
        "I build backend APIs with node and mongo, and build UI with reactjs. "
        "Also working on ml and nlp projects."
    )
    result = extractor.extract(text)

    assert "Node.js" in result["skills"]
    assert "MongoDB" in result["skills"]
    assert "React" in result["skills"]
    assert "Machine Learning" in result["skills"]
    assert "NLP" in result["skills"]

    text_js = "Worked with js."
    result_js = extractor.extract(text_js)
    assert "JavaScript" in result_js["skills"]


def test_api_route_with_skills():
    """POST /resume/parse should return extracted skills from a PDF resume."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "Jane Doe\n\nSKILLS\nFastAPI, js, MongoDB, Docker, DSA\n\nEXPERIENCE\nML Engineer Intern",
    )
    pdf_bytes = doc.write()
    doc.close()

    response = client.post(
        "/api/v1/resume/parse",
        files={"file": ("test_skills_resume.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    extracted = data["skills"]
    assert "FastAPI" in extracted
    assert "JavaScript" in extracted
    assert "MongoDB" in extracted
    assert "Docker" in extracted
    assert "DSA" in extracted
    assert "Machine Learning" in extracted
