"""
Tests for ML model management endpoints:
  - GET  /api/v1/recommend/ml-model/status
  - POST /api/v1/recommend/ml-model/train
  - GET  /api/v1/recommend/ml-model/download
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.main import app
from app.services.fit_classifier import train_and_save_model, MODEL_PATH

client = TestClient(app)

STATUS_URL = "/api/v1/recommend/ml-model/status"
TRAIN_URL = "/api/v1/recommend/ml-model/train"
DOWNLOAD_URL = "/api/v1/recommend/ml-model/download"


@pytest.fixture(autouse=True)
def ensure_model_trained():
    """Guarantee the classifier pkl exists before every test in this module."""
    train_and_save_model()


# ---------------------------------------------------------------------------
# GET /ml-model/status
# ---------------------------------------------------------------------------

class TestMLModelStatus:
    def test_status_returns_200(self):
        response = client.get(STATUS_URL)
        assert response.status_code == 200

    def test_status_has_required_keys(self):
        data = client.get(STATUS_URL).json()
        assert "model_loaded" in data
        assert "metrics" in data
        assert "feature_names" in data
        assert "feature_importances" in data

    def test_status_model_loaded_true(self):
        data = client.get(STATUS_URL).json()
        assert data["model_loaded"] is True

    def test_status_metrics_have_accuracy(self):
        data = client.get(STATUS_URL).json()
        metrics = data["metrics"]
        assert metrics is not None
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_status_metrics_have_f1(self):
        data = client.get(STATUS_URL).json()
        assert "f1_score" in data["metrics"]

    def test_status_feature_importances_is_dict(self):
        data = client.get(STATUS_URL).json()
        fi = data["feature_importances"]
        assert isinstance(fi, dict)
        assert len(fi) > 0

    def test_status_feature_importances_sum_approx_one(self):
        data = client.get(STATUS_URL).json()
        total = sum(data["feature_importances"].values())
        assert abs(total - 1.0) < 0.01, f"Importances should sum ~1.0 but got {total}"

    def test_status_feature_names_match_importances(self):
        data = client.get(STATUS_URL).json()
        assert set(data["feature_names"]) == set(data["feature_importances"].keys())


# ---------------------------------------------------------------------------
# POST /ml-model/train
# ---------------------------------------------------------------------------

class TestMLModelTrain:
    def test_train_returns_200(self):
        response = client.post(TRAIN_URL)
        assert response.status_code == 200

    def test_train_returns_message(self):
        data = client.post(TRAIN_URL).json()
        assert "message" in data
        assert data["message"] != ""

    def test_train_returns_metrics(self):
        data = client.post(TRAIN_URL).json()
        assert "metrics" in data
        metrics = data["metrics"]
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_train_accuracy_reasonable(self):
        data = client.post(TRAIN_URL).json()
        assert data["metrics"]["accuracy"] >= 0.5, "Accuracy should be at least 50%"

    def test_train_f1_reasonable(self):
        data = client.post(TRAIN_URL).json()
        assert data["metrics"]["f1_score"] >= 0.5, "F1 score should be at least 50%"

    def test_model_file_exists_after_train(self):
        client.post(TRAIN_URL)
        assert os.path.exists(MODEL_PATH), "Model pkl should exist after retraining"

    def test_train_followed_by_status_shows_loaded(self):
        client.post(TRAIN_URL)
        status_data = client.get(STATUS_URL).json()
        assert status_data["model_loaded"] is True


# ---------------------------------------------------------------------------
# GET /ml-model/download
# ---------------------------------------------------------------------------

class TestMLModelDownload:
    def test_download_returns_200(self):
        response = client.get(DOWNLOAD_URL)
        assert response.status_code == 200

    def test_download_content_type_is_octet_stream(self):
        response = client.get(DOWNLOAD_URL)
        assert "octet-stream" in response.headers.get("content-type", "")

    def test_download_has_content(self):
        response = client.get(DOWNLOAD_URL)
        assert len(response.content) > 0

    def test_download_content_is_valid_pickle(self):
        import io
        import joblib
        response = client.get(DOWNLOAD_URL)
        # Model was saved with joblib — load it back from bytes
        bundle = joblib.load(io.BytesIO(response.content))
        assert "model" in bundle
        assert "feature_names" in bundle



# ---------------------------------------------------------------------------
# Recommendation endpoint with use_ml flag
# ---------------------------------------------------------------------------

class TestRecommendWithMLToggle:
    """Verify the use_ml flag is respected in the recommendations endpoint."""

    SAMPLE_JOB = {
        "source": "Test",
        "title": "Python Backend Engineer",
        "company": "AcmeCorp",
        "location": "Remote",
        "description": "Backend APIs with Python and FastAPI.",
        "requirements": "Python, FastAPI, SQL, Git, Docker",
        "apply_url": "https://example.com/apply",
        "job_type": "Full-time",
    }

    PAYLOAD_BASE = {
        "resume_text": "Software engineer with Python, FastAPI, SQL, Git, and Docker projects.",
        "resume_skills": ["Python", "FastAPI", "SQL", "Git", "Docker"],
        "jobs": [SAMPLE_JOB],
    }

    def test_recommend_with_ml_true(self):
        payload = {**self.PAYLOAD_BASE, "use_ml": True}
        response = client.post("/api/v1/recommend/jobs", json=payload)
        assert response.status_code == 200
        recs = response.json()["recommendations"]
        assert len(recs) == 1
        assert recs[0]["fit_label"] in ["Low Fit", "Medium Fit", "High Fit"]

    def test_recommend_with_ml_false(self):
        payload = {**self.PAYLOAD_BASE, "use_ml": False}
        response = client.post("/api/v1/recommend/jobs", json=payload)
        assert response.status_code == 200
        recs = response.json()["recommendations"]
        assert len(recs) == 1
        assert recs[0]["fit_label"] in ["Low Fit", "Medium Fit", "High Fit"]

    def test_recommend_default_uses_ml(self):
        # use_ml should default to True — both should work
        payload = dict(self.PAYLOAD_BASE)  # no use_ml key
        response = client.post("/api/v1/recommend/jobs", json=payload)
        assert response.status_code == 200
        assert len(response.json()["recommendations"]) == 1
