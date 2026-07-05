"""Tests for job text normalization and common job-board output format."""

from app.services.job_sources.normalizer import (
    clean_html,
    parse_job_type,
    split_description_and_requirements,
)
from app.services.job_sources import fetch_jobs
from tests.helpers import assert_all_common_job_format, assert_common_job_format, COMMON_JOB_FIELDS
from unittest.mock import MagicMock, patch


def test_clean_html_strips_tags_and_entities():
    """HTML job descriptions should become readable plain text."""
    raw = "<p>Build APIs with <strong>Python</strong>.</p><br/>Requirements: Docker &amp; SQL"
    cleaned = clean_html(raw)
    assert "Python" in cleaned
    assert "Docker & SQL" in cleaned
    assert "<" not in cleaned


def test_split_description_and_requirements():
    """Requirement headings should split description from qualifications."""
    text = (
        "We are hiring a backend engineer.\n\n"
        "Requirements:\nPython, FastAPI, and PostgreSQL experience."
    )
    description, requirements = split_description_and_requirements(text)
    assert "backend engineer" in description
    assert "Python" in requirements
    assert "Requirements" not in requirements


def test_parse_job_type_detects_intern_and_fulltime():
    """Job type heuristics should detect internship and full-time roles."""
    assert parse_job_type("Software Engineer Intern", "") == "Internship"
    assert parse_job_type("Backend Developer", "This is a full-time role.") == "Full-time"
    assert parse_job_type("Contract Role", "Short project") == "Unknown"


@patch("requests.get")
def test_greenhouse_response_uses_common_format(mock_get: MagicMock):
    """Greenhouse API payloads should normalize into the shared job schema."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "jobs": [{
                "title": "Data Engineer",
                "location": {"name": "Austin, TX"},
                "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
                "content": "<p>Build pipelines.</p><p>Requirements: Python and SQL</p>",
            }]
        },
    )
    jobs = fetch_jobs("greenhouse", "acme")
    assert_all_common_job_format(jobs)
    assert jobs[0]["source"] == "Greenhouse"


@patch("requests.get")
def test_lever_response_uses_common_format(mock_get: MagicMock):
    """Lever API payloads should normalize into the shared job schema."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: [{
            "title": "Platform Engineer",
            "hostedUrl": "https://jobs.lever.co/acme/1",
            "categories": {"location": "Remote", "commitment": "Full-time"},
            "descriptionHtml": "<p>Platform team.</p>",
            "lists": [{"text": "Requirements", "content": "<ul><li>Go</li></ul>"}],
        }],
    )
    jobs = fetch_jobs("lever", "acme")
    assert_all_common_job_format(jobs)
    assert jobs[0]["source"] == "Lever"


@patch("requests.get")
def test_ashby_response_uses_common_format(mock_get: MagicMock):
    """Ashby API payloads should normalize into the shared job schema."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "jobs": [{
                "title": "ML Engineer",
                "location": "Boston, MA",
                "employmentType": "FullTime",
                "jobUrl": "https://jobs.ashbyhq.com/acme/1",
                "descriptionPlain": "Train models.\n\nRequirements: PyTorch and Python",
            }]
        },
    )
    jobs = fetch_jobs("ashby", "acme")
    assert_all_common_job_format(jobs)
    assert jobs[0]["source"] == "Ashby"


def test_common_job_fields_constant_documents_schema():
    """Document the expected normalized job keys for all connectors."""
    sample = {field: "value" for field in COMMON_JOB_FIELDS}
    assert_common_job_format(sample)
