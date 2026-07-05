"""Tests for resume parsing: text cleaning, sectioning, and PDF API."""

import fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.resume_parser import clean_text, parse_resume, segment_sections

client = TestClient(app)


def test_clean_text():
    """Tabs, extra spaces, and separator lines should be normalized."""
    dirty_text = "Python\t\tDeveloper\n\n\n\nExperience   at   Company\n-------\n"
    expected = "Python Developer\n\nExperience at Company"
    assert clean_text(dirty_text) == expected


def test_segment_sections():
    """Standard resume headings should populate skills, experience, and education."""
    sample_text = (
        "John Doe\n"
        "Email: john@example.com\n\n"
        "SKILLS\n"
        "Python, FastAPI, SQL\n\n"
        "EXPERIENCE\n"
        "Software Engineer at TechCorp\n"
        "- Built APIs and web scrapers\n\n"
        "EDUCATION\n"
        "BS in Computer Science\n"
    )

    sections = segment_sections(sample_text)

    assert "Python, FastAPI, SQL" in sections["skills"]
    assert "Software Engineer at TechCorp" in sections["experience"]
    assert "BS in Computer Science" in sections["education"]
    assert sections["projects"] == ""
    assert sections["certifications"] == ""


def test_parse_resume_from_in_memory_pdf():
    """parse_resume should return cleaned text and structured sections from a PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "Alex Cooper\n\nSKILLS\nPython, FastAPI, React\n\nPROJECTS\nBuilt REST API\n\nEDUCATION\nBS Computer Science",
    )
    pdf_bytes = doc.write()
    doc.close()

    result = parse_resume(pdf_bytes)

    assert "raw_text" in result
    assert "sections" in result
    assert "Python" in result["sections"]["skills"]
    assert "REST API" in result["sections"]["projects"]
    assert "Computer Science" in result["sections"]["education"]


def test_api_parse_invalid_file_type():
    """Non-PDF uploads should return a clear 400 error."""
    response = client.post(
        "/api/v1/resume/parse",
        files={"file": ("resume.txt", b"Some random plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_api_parse_empty_file():
    """Empty PDF uploads should be rejected."""
    response = client.post(
        "/api/v1/resume/parse",
        files={"file": ("resume.pdf", b"", "application/pdf")},
    )
    assert response.status_code in (400, 422)


def test_api_parse_valid_pdf():
    """End-to-end PDF upload should return sections and extracted skills."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "Jane Doe\n\nSKILLS\nMachine Learning, Streamlit\n\nEDUCATION\nUniversity of Waterloo",
    )
    pdf_bytes = doc.write()
    doc.close()

    response = client.post(
        "/api/v1/resume/parse",
        files={"file": ("test_resume.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "raw_text" in data
    assert "sections" in data
    assert "Machine Learning" in data["sections"]["skills"]
    assert "Waterloo" in data["sections"]["education"]
    assert data["skill_count"] >= 1


def test_api_parse_text_resume():
    """Pasted resume text should parse with the same shape as PDF uploads."""
    response = client.post(
        "/api/v1/resume/parse-text",
        json={
            "filename": "pasted.txt",
            "resume_text": (
                "Alex Cooper\n\n"
                "SKILLS\nPython, FastAPI, SQL\n\n"
                "PROJECTS\nBuilt a job matching API"
            ),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "raw_text" in data
    assert "sections" in data
    assert "Python" in data["sections"]["skills"]
    assert data["skill_count"] >= 1
