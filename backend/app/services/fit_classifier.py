"""
Supervised job-fit classifier (Low / Medium / High Fit).

Trained on a synthetic resume-job dataset for MVP. Replace with real
application-outcome labels when available.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "skill_match_score",
    "semantic_similarity_score",
    "number_of_matched_skills",
    "number_of_missing_skills",
    "has_project_keywords",
    "has_internship_keywords",
]

FIT_LABELS = ["Low Fit", "Medium Fit", "High Fit"]

PROJECT_KEYWORD_PATTERN = re.compile(
    r"\b(project|projects|built|developed|deployed|github|portfolio|capstone)\b",
    re.IGNORECASE,
)
INTERNSHIP_KEYWORD_PATTERN = re.compile(
    r"\b(intern|internship|student|campus|graduate|new grad|entry level)\b",
    re.IGNORECASE,
)

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(_MODEL_DIR, "fit_classifier.pkl")
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

_classifier_bundle: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def has_project_keywords(text: str) -> int:
    return int(bool(PROJECT_KEYWORD_PATTERN.search(text or "")))


def has_internship_keywords(text: str) -> int:
    return int(bool(INTERNSHIP_KEYWORD_PATTERN.search(text or "")))


def build_feature_vector(
    resume_text: str,
    skill_match_score: float,
    semantic_similarity_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
) -> Dict[str, float]:
    """Build a feature dict for one resume-job pair."""
    return {
        "skill_match_score": float(skill_match_score),
        "semantic_similarity_score": float(semantic_similarity_score),
        "number_of_matched_skills": float(len(matched_skills)),
        "number_of_missing_skills": float(len(missing_skills)),
        "has_project_keywords": float(has_project_keywords(resume_text)),
        "has_internship_keywords": float(has_internship_keywords(resume_text)),
    }


def features_to_array(features: Dict[str, float]) -> np.ndarray:
    return np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)


def heuristic_fit_label(features: Dict[str, float]) -> str:
    """Rule-based fallback when the ML model is unavailable."""
    composite = (
        0.40 * features["skill_match_score"]
        + 0.35 * features["semantic_similarity_score"]
        + 0.10 * min(features["number_of_matched_skills"] / 5.0, 1.0)
        - 0.08 * min(features["number_of_missing_skills"] / 5.0, 1.0)
        + 0.05 * features["has_project_keywords"]
        + 0.04 * features["has_internship_keywords"]
    )
    if composite >= 0.62:
        return "High Fit"
    if composite <= 0.35:
        return "Low Fit"
    return "Medium Fit"


# ---------------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------------

SYNTHETIC_RESUMES: List[Dict[str, Any]] = [
    {
        "text": "Software Engineering Intern seeking internship. Skills: Python, FastAPI, React, SQL, Git. "
        "Projects: Built REST API and React dashboard deployed on Docker.",
        "skills": ["Python", "FastAPI", "React", "SQL", "Git", "Docker"],
    },
    {
        "text": "Machine Learning student with projects in NLP and semantic search. "
        "Skills: Python, scikit-learn, spaCy, SQL, FastAPI, Git, REST API.",
        "skills": ["Python", "Scikit-learn", "spaCy", "SQL", "FastAPI", "Git", "REST API", "NLP", "Machine Learning"],
    },
    {
        "text": "AI Product Engineer profile. Python, Streamlit, React, PyMuPDF, Docker, NLP, Machine Learning. "
        "Developed Streamlit apps and deployed ML demos.",
        "skills": ["Python", "Streamlit", "React", "PyMuPDF", "Docker", "NLP", "Machine Learning"],
    },
    {
        "text": "Junior developer with JavaScript and HTML only. No major projects listed.",
        "skills": ["JavaScript", "HTML", "CSS"],
    },
    {
        "text": "Data analyst graduate. SQL, Python, Pandas, Excel. Completed analytics capstone project.",
        "skills": ["SQL", "Python", "Pandas", "NumPy"],
    },
    {
        "text": "Backend engineer. Python, FastAPI, PostgreSQL, Docker, AWS, Git. "
        "Built microservices and database-backed APIs.",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "Git", "SQL"],
    },
    {
        "text": "Computer Science student applying for campus internship. "
        "Learning C++ and basic algorithms. No deployed projects yet.",
        "skills": ["C++", "Java"],
    },
    {
        "text": "Full stack intern candidate. Python, Flask, React, JavaScript, TypeScript, PostgreSQL, Docker.",
        "skills": ["Python", "Flask", "React", "JavaScript", "TypeScript", "PostgreSQL", "Docker"],
    },
    {
        "text": "ML engineer with TensorFlow, PyTorch, Python, NLP, Deep Learning. "
        "GitHub portfolio with deployed Streamlit and FastAPI models.",
        "skills": ["Python", "TensorFlow", "PyTorch", "NLP", "Deep Learning", "Machine Learning", "FastAPI", "Streamlit"],
    },
    {
        "text": "Marketing graduate exploring tech roles. Basic Excel and communication skills.",
        "skills": [],
    },
]


def _simple_semantic_similarity(resume_text: str, job_text: str) -> float:
    """Lightweight token overlap proxy used only for synthetic dataset generation."""
    resume_tokens = set(re.findall(r"[a-zA-Z0-9+#.]+", resume_text.lower()))
    job_tokens = set(re.findall(r"[a-zA-Z0-9+#.]+", job_text.lower()))
    if not resume_tokens or not job_tokens:
        return 0.0
    overlap = len(resume_tokens & job_tokens)
    return min(1.0, overlap / max(len(job_tokens), 1))


def _load_sample_jobs() -> List[Dict[str, Any]]:
    path = os.path.join(_DATA_DIR, "sample_jobs.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compute_skill_overlap(resume_skills: List[str], job_text: str) -> Tuple[float, List[str], List[str]]:
    from app.services.skill_extractor import extractor

    job_skills = extractor.extract(job_text).get("skills", [])
    resume_map = {s.lower(): s for s in resume_skills}
    job_map = {s.lower(): s for s in job_skills}
    matched = sorted(resume_map[k] for k in resume_map if k in job_map)
    missing = sorted(job_map[k] for k in job_map if k not in resume_map)
    if not job_map:
        return 1.0, matched, missing
    return len(matched) / len(job_map), matched, missing


def _build_job_text(job: Dict[str, Any]) -> str:
    return " ".join(
        job.get(field, "")
        for field in ("title", "description", "requirements", "location")
        if job.get(field)
    )


def assign_training_label(features: Dict[str, float]) -> str:
    """Create synthetic supervision labels from feature combinations."""
    return heuristic_fit_label(features)


def generate_synthetic_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build labeled training rows from sample jobs and synthetic resumes.
    Also adds noisy variants to increase dataset size.
    """
    jobs = _load_sample_jobs()
    rows: List[List[float]] = []
    labels: List[str] = []

    extra_jobs = [
        {
            "title": "Python Developer",
            "description": "Backend Python services with SQL and Git.",
            "requirements": "Python, SQL, Git, Docker",
            "location": "Remote",
        },
        {
            "title": "Data Science Intern",
            "description": "Internship for students with ML and pandas experience.",
            "requirements": "Python, Pandas, SQL, Machine Learning, scikit-learn",
            "location": "India",
        },
        {
            "title": "Frontend Engineer",
            "description": "React and TypeScript UI development.",
            "requirements": "React, JavaScript, TypeScript, HTML, CSS",
            "location": "Remote",
        },
    ]
    all_jobs = jobs + extra_jobs

    for resume in SYNTHETIC_RESUMES:
        for job in all_jobs:
            job_text = _build_job_text(job)
            skill_match, matched, missing = _compute_skill_overlap(resume["skills"], job_text)
            semantic = _simple_semantic_similarity(resume["text"], job_text)
            features = build_feature_vector(
                resume["text"], skill_match, semantic, matched, missing
            )
            rows.append([features[name] for name in FEATURE_NAMES])
            labels.append(assign_training_label(features))

            # Noisy variant: slightly worse semantic score
            noisy = features.copy()
            noisy["semantic_similarity_score"] = max(0.0, semantic - 0.15)
            rows.append([noisy[name] for name in FEATURE_NAMES])
            labels.append(assign_training_label(noisy))

    return np.array(rows, dtype=float), np.array(labels)


# ---------------------------------------------------------------------------
# Train / load / predict
# ---------------------------------------------------------------------------

def train_model(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Random Forest classifier and return evaluation metrics."""
    x, y = generate_synthetic_training_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
    }

    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "fit_labels": FIT_LABELS,
        "metrics": metrics,
    }
    return bundle


def save_model(bundle: Dict[str, Any], path: str = MODEL_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Saved fit classifier to %s", path)
    return path


def load_model(path: str = MODEL_PATH) -> Optional[Dict[str, Any]]:
    global _classifier_bundle
    if _classifier_bundle is not None:
        return _classifier_bundle
    if not os.path.exists(path):
        return None
    _classifier_bundle = joblib.load(path)
    return _classifier_bundle


def predict_fit_label(
    resume_text: str,
    skill_match_score: float,
    semantic_similarity_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
) -> str:
    """Predict Low / Medium / High Fit for a resume-job pair."""
    features = build_feature_vector(
        resume_text,
        skill_match_score,
        semantic_similarity_score,
        matched_skills,
        missing_skills,
    )

    bundle = load_model()
    if bundle is None:
        return heuristic_fit_label(features)

    model = bundle["model"]
    prediction = model.predict(features_to_array(features))[0]
    if prediction in FIT_LABELS:
        return str(prediction)
    return heuristic_fit_label(features)


def train_and_save_model() -> Dict[str, Any]:
    """Train, evaluate, and persist the classifier."""
    bundle = train_model()
    save_model(bundle)
    return bundle["metrics"]


if __name__ == "__main__":
    metrics = train_and_save_model()
    print("Fit classifier trained and saved.")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(metrics["classification_report"])
