"""
Job matching engine: ranks jobs against a resume using skill overlap,
semantic similarity, and role/title keyword bonuses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.services.skill_extractor import extractor
from app.services.fit_classifier import predict_fit_label

logger = logging.getLogger(__name__)

# Role keywords used for the title bonus (matched against job title + resume text)
ROLE_KEYWORDS = [
    "machine learning",
    "ai",
    "data analyst",
    "full stack",
    "backend",
    "frontend",
    "software engineer",
    "intern",
]

# Scoring weights
WEIGHT_SKILL = 0.45
WEIGHT_SEMANTIC = 0.45
WEIGHT_TITLE = 0.10

_st_model = None
_st_util = None
_use_tfidf_fallback = False


def _load_sentence_transformer():
    """Try to load sentence-transformers; fall back to TF-IDF on failure."""
    global _st_model, _st_util, _use_tfidf_fallback

    if _st_model is not None or _use_tfidf_fallback:
        return

    if not settings.ENABLE_SENTENCE_TRANSFORMERS:
        logger.info("SentenceTransformers disabled. Using TF-IDF semantic fallback.")
        _use_tfidf_fallback = True
        return

    try:
        from sentence_transformers import SentenceTransformer, util

        logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        _st_util = util
        logger.info("SentenceTransformer model loaded successfully.")
    except Exception as exc:
        logger.warning(
            "sentence-transformers unavailable (%s). Using TF-IDF fallback.", exc
        )
        _use_tfidf_fallback = True


def _semantic_similarity(resume_text: str, job_text: str) -> float:
    """Return semantic similarity in [0, 1] between resume and job text."""
    scores = _semantic_similarity_batch(resume_text, [job_text])
    return scores[0] if scores else 0.0


def _semantic_similarity_batch(resume_text: str, job_texts: List[str]) -> List[float]:
    """Return semantic similarity scores in [0, 1] for many job texts."""
    if not job_texts:
        return []

    _load_sentence_transformer()

    if _st_model is not None and _st_util is not None:
        resume_emb = _st_model.encode(resume_text, convert_to_tensor=True)
        job_embs = _st_model.encode(job_texts, convert_to_tensor=True)
        cosine_scores = _st_util.cos_sim(resume_emb, job_embs)[0]
        return [max(0.0, min(1.0, float(score))) for score in cosine_scores]

    # TF-IDF + cosine similarity fallback
    vectorizer = TfidfVectorizer(stop_words="english")
    documents = [resume_text or ""] + [job_text or "" for job_text in job_texts]
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return [0.0 for _ in job_texts]

    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return [max(0.0, min(1.0, float(score))) for score in similarities]


def _build_job_text(job: Dict[str, Any]) -> str:
    """Combine job fields into one searchable text block."""
    parts = [
        job.get("title", ""),
        job.get("description", ""),
        job.get("requirements", ""),
        job.get("location", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _normalize_skill_map(skills: List[str]) -> Dict[str, str]:
    """Map lowercase skill names to their display form."""
    return {s.lower(): s for s in skills if s}


def _compute_skill_match(
    resume_skills: List[str], job_skills: List[str]
) -> tuple[float, List[str], List[str]]:
    """
    Skill match = matched resume skills / total required job skills.
    Returns (score, matched_skills, missing_skills).
    """
    resume_map = _normalize_skill_map(resume_skills)
    job_map = _normalize_skill_map(job_skills)

    matched = sorted(resume_map[key] for key in resume_map if key in job_map)
    missing = sorted(job_map[key] for key in job_map if key not in resume_map)

    if not job_map:
        return 1.0, matched, missing

    score = len(matched) / len(job_map)
    return score, matched, missing


def _compute_title_bonus(title: str, resume_text: str, projects_text: str = "") -> float:
    """
    Small bonus when role keywords appear in the job title and in resume/projects.
    """
    title_lower = title.lower()
    searchable = f"{resume_text} {projects_text}".lower()

    for keyword in ROLE_KEYWORDS:
        if keyword in title_lower and keyword in searchable:
            return 1.0
    return 0.0


def _matched_role_label(title: str, resume_text: str, projects_text: str = "") -> Optional[str]:
    """Return a human-readable role label when title bonus applies."""
    title_lower = title.lower()
    searchable = f"{resume_text} {projects_text}".lower()

    labels = {
        "machine learning": "Machine Learning",
        "ai": "AI",
        "data analyst": "Data Analyst",
        "full stack": "Full Stack",
        "backend": "Backend",
        "frontend": "Frontend",
        "software engineer": "Software Engineer",
        "intern": "Intern",
    }
    for keyword, label in labels.items():
        if keyword in title_lower and keyword in searchable:
            return label
    return None


def _build_reason(
    matched_skills: List[str],
    title: str,
    resume_text: str,
    projects_text: str = "",
) -> str:
    """Generate a short explanation for the recommendation."""
    if matched_skills:
        skills_part = ", ".join(matched_skills[:4])
        if len(matched_skills) > 4:
            skills_part += f" (+{len(matched_skills) - 4} more)"
        reason = f"Matched because your resume contains {skills_part}"
    else:
        reason = "Matched based on semantic profile overlap"

    role_label = _matched_role_label(title, resume_text, projects_text)
    if role_label:
        reason += f" and project experience relevant to this {role_label} role."
    else:
        reason += " and experience relevant to this role."

    return reason


def rank_jobs(
    resume_text: str,
    resume_skills: List[str],
    jobs: List[Dict[str, Any]],
    projects_text: str = "",
    use_ml: bool = True,
) -> List[Dict[str, Any]]:
    """
    Rank jobs by relevance to the candidate resume.

    Formula:
      final_score = 0.45 * skill_match + 0.45 * semantic_similarity + 0.10 * title_bonus
    """
    if not jobs:
        return []

    ranked_results: List[Dict[str, Any]] = []
    job_texts = [_build_job_text(job) for job in jobs]
    semantic_scores = _semantic_similarity_batch(resume_text, job_texts)

    for job, job_text, semantic_similarity_score in zip(jobs, job_texts, semantic_scores):
        title = job.get("title", "Unknown Title")
        company = job.get("company", "Unknown Company")
        location = job.get("location", "Remote/Unknown")
        apply_url = job.get("apply_url", "#")
        source = job.get("source", "Unknown")

        job_skills = extractor.extract(job_text).get("skills", [])
        skill_match_score, matched_skills, missing_skills = _compute_skill_match(
            resume_skills, job_skills
        )

        title_bonus = _compute_title_bonus(title, resume_text, projects_text)

        final_score = (
            WEIGHT_SKILL * skill_match_score
            + WEIGHT_SEMANTIC * semantic_similarity_score
            + WEIGHT_TITLE * title_bonus
        )
        match_score = int(min(1.0, max(0.0, final_score)) * 100)

        if use_ml:
            fit_label = predict_fit_label(
                resume_text=resume_text,
                skill_match_score=skill_match_score,
                semantic_similarity_score=semantic_similarity_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
            )
        else:
            from app.services.fit_classifier import build_feature_vector, heuristic_fit_label
            features = build_feature_vector(
                resume_text=resume_text,
                skill_match_score=skill_match_score,
                semantic_similarity_score=semantic_similarity_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
            )
            fit_label = heuristic_fit_label(features)

        ranked_results.append(
            {
                "title": title,
                "company": company,
                "location": location,
                "apply_url": apply_url,
                "source": source,
                "match_score": match_score,
                "fit_label": fit_label,
                "skill_match_score": round(skill_match_score, 4),
                "semantic_similarity_score": round(semantic_similarity_score, 4),
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "reason": _build_reason(
                    matched_skills, title, resume_text, projects_text
                ),
            }
        )

    ranked_results.sort(key=lambda item: item["match_score"], reverse=True)
    return ranked_results
