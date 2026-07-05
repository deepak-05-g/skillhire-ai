"""Tests for Greenhouse, Lever, and Ashby job source connectors."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.job_sources import fetch_jobs
from tests.helpers import assert_all_common_job_format

client = TestClient(app)

@patch("requests.get")
def test_fetch_greenhouse_jobs(mock_get):
    """
    Test Greenhouse connector fetch and normalization.
    """
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jobs": [
            {
                "title": "Software Engineer Intern",
                "location": {"name": "SF, CA"},
                "absolute_url": "https://greenhouse.io/test/123",
                "content": "<p>We are looking for an intern. Requirements: Python, SQL, and Docker.</p>"
            }
        ]
    }
    mock_get.return_value = mock_response
    
    jobs = fetch_jobs("greenhouse", "testcompany")

    assert len(jobs) == 1
    assert_all_common_job_format(jobs)
    job = jobs[0]
    assert job["source"] == "Greenhouse"
    assert job["company"] == "Testcompany"
    assert job["title"] == "Software Engineer Intern"
    assert job["location"] == "SF, CA"
    assert "intern" in job["description"].lower()
    assert "Python, SQL, and Docker" in job["requirements"]
    assert job["apply_url"] == "https://greenhouse.io/test/123"
    assert job["job_type"] == "Internship"

@patch("requests.get")
def test_fetch_lever_jobs(mock_get):
    """
    Test Lever connector fetch and normalization.
    """
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "title": "Backend Developer",
            "hostedUrl": "https://lever.co/test/456",
            "categories": {
                "location": "Remote",
                "commitment": "Full-time"
            },
            "descriptionHtml": "<p>Great company.</p>",
            "lists": [
                {
                    "text": "Requirements",
                    "content": "<ul><li>Go and Python</li></ul>"
                }
            ]
        }
    ]
    mock_get.return_value = mock_response
    
    jobs = fetch_jobs("lever", "testcompany")

    assert len(jobs) == 1
    assert_all_common_job_format(jobs)
    job = jobs[0]
    assert job["source"] == "Lever"
    assert job["company"] == "Testcompany"
    assert job["title"] == "Backend Developer"
    assert job["location"] == "Remote"
    assert "Go and Python" in job["requirements"]
    assert job["job_type"] == "Full-time"

@patch("requests.get")
def test_fetch_ashby_jobs(mock_get):
    """
    Test Ashby connector fetch and normalization.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jobs": [
            {
                "title": "Machine Learning Engineer",
                "location": "Boston, MA",
                "employmentType": "FullTime",
                "jobUrl": "https://ashbyhq.com/test/789",
                "descriptionPlain": "We build models. Requirements: PyTorch, ML, and Statistics."
            }
        ]
    }
    mock_get.return_value = mock_response
    
    jobs = fetch_jobs("ashby", "testcompany")

    assert len(jobs) == 1
    assert_all_common_job_format(jobs)
    job = jobs[0]
    assert job["source"] == "Ashby"
    assert job["company"] == "Testcompany"
    assert job["title"] == "Machine Learning Engineer"
    assert job["location"] == "Boston, MA"
    assert "PyTorch, ML, and Statistics" in job["requirements"]
    assert job["job_type"] == "Full-time"

@patch("requests.get")
def test_fetch_jobs_endpoint(mock_get):
    """
    Test the GET /api/v1/jobs/fetch route using FastAPI test client.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"jobs": []}
    mock_get.return_value = mock_response
    
    response = client.get("/api/v1/jobs/fetch?source=greenhouse&company=stripe")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_fetch_jobs_invalid_source():
    """
    Test route validations for invalid source parameters.
    """
    response = client.get("/api/v1/jobs/fetch?source=invalidboard&company=stripe")
    assert response.status_code == 400
    assert "Invalid source" in response.json()["detail"]
