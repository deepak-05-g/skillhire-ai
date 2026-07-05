import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.fit_classifier import (
    FIT_LABELS,
    FEATURE_NAMES,
    build_feature_vector,
    generate_synthetic_training_data,
    heuristic_fit_label,
    load_model,
    predict_fit_label,
    train_and_save_model,
)
from app.services.matcher import rank_jobs


def test_synthetic_training_data_shape():
    """Synthetic dataset should produce rows for every resume-job combination."""
    x, y = generate_synthetic_training_data()
    assert x.shape[1] == len(FEATURE_NAMES)
    assert len(y) == len(x)
    assert len(x) >= 20
    assert set(y).issubset(set(FIT_LABELS))


def test_train_and_save_model():
    """Training should persist a loadable classifier bundle."""
    metrics = train_and_save_model()
    assert metrics["accuracy"] >= 0.5
    assert metrics["f1_score"] >= 0.5

    bundle = load_model()
    assert bundle is not None
    assert bundle["model"] is not None


def test_predict_fit_label_returns_valid_label():
    """Predictions should be one of the three fit labels."""
    train_and_save_model()
    label = predict_fit_label(
        resume_text="Software intern with Python projects and GitHub portfolio.",
        skill_match_score=0.8,
        semantic_similarity_score=0.75,
        matched_skills=["Python", "FastAPI", "Git"],
        missing_skills=["Docker"],
    )
    assert label in FIT_LABELS


@patch("app.services.matcher._semantic_similarity", return_value=0.8)
def test_matcher_includes_fit_label(mock_semantic):
    """Ranked jobs should include ML fit_label field."""
    train_and_save_model()
    job = {
        "source": "Greenhouse",
        "title": "Backend Engineer",
        "company": "TechCo",
        "location": "Remote",
        "description": "Python FastAPI backend role.",
        "requirements": "Python, FastAPI, SQL, Git",
        "apply_url": "https://example.com/job",
        "job_type": "Full-time",
    }
    results = rank_jobs(
        "Software engineer with Python, FastAPI, SQL projects.",
        ["Python", "FastAPI", "SQL", "Git"],
        [job],
    )
    assert results[0]["fit_label"] in FIT_LABELS


def test_heuristic_fallback_when_model_missing(monkeypatch, tmp_path):
    """Without a saved model, heuristic labels should still work."""
    missing_path = tmp_path / "missing.pkl"
    monkeypatch.setattr(
        "app.services.fit_classifier.MODEL_PATH",
        str(missing_path),
    )
    monkeypatch.setattr("app.services.fit_classifier._classifier_bundle", None)

    features = build_feature_vector(
        "Intern with projects",
        skill_match_score=0.2,
        semantic_similarity_score=0.2,
        matched_skills=["Python"],
        missing_skills=["Docker", "AWS", "React", "SQL"],
    )
    assert heuristic_fit_label(features) == "Low Fit"
