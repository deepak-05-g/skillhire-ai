"""
Database persistence helpers for resumes, jobs, and recommendations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models import Job, Recommendation, Resume, SavedJob


def _job_to_dict(record: Job) -> Dict[str, Any]:
    return {
        "id": record.id,
        "source": record.source,
        "company": record.company,
        "title": record.title,
        "location": record.location,
        "description": record.description,
        "requirements": record.requirements,
        "apply_url": record.apply_url,
        "job_type": record.job_type,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


def save_parsed_resume(
    db: Session,
    filename: str,
    raw_text: str,
    extracted_skills: List[str],
) -> Resume:
    """Persist a parsed resume record."""
    record = Resume(
        filename=filename,
        raw_text=raw_text,
        extracted_skills=extracted_skills,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def _find_existing_job(db: Session, job_data: Dict[str, Any]) -> Optional[Job]:
    """Look up a job by apply_url or title/company/source tuple."""
    apply_url = job_data.get("apply_url", "").strip()
    if apply_url and apply_url != "#":
        existing = db.query(Job).filter(Job.apply_url == apply_url).first()
        if existing:
            return existing

    return (
        db.query(Job)
        .filter(
            Job.title == job_data.get("title", ""),
            Job.company == job_data.get("company", ""),
            Job.source == job_data.get("source", "Unknown"),
        )
        .first()
    )


def upsert_job(db: Session, job_data: Dict[str, Any]) -> Job:
    """Insert a job or return an existing matching record."""
    existing = _find_existing_job(db, job_data)
    if existing:
        return existing

    job = Job(
        source=job_data.get("source", "Unknown"),
        company=job_data.get("company", "Unknown"),
        title=job_data.get("title", "Unknown Title"),
        location=job_data.get("location", ""),
        description=job_data.get("description", ""),
        requirements=job_data.get("requirements", ""),
        apply_url=job_data.get("apply_url", ""),
        job_type=job_data.get("job_type", "Unknown"),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def save_jobs_batch(db: Session, jobs: List[Dict[str, Any]]) -> List[Job]:
    """Persist a list of normalized job dicts."""
    return [upsert_job(db, job) for job in jobs]


def list_jobs(db: Session, limit: int = 500) -> List[Dict[str, Any]]:
    """Return stored jobs newest first."""
    rows = (
        db.query(Job)
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(limit)
        .all()
    )
    return [_job_to_dict(row) for row in rows]


def get_or_create_resume(
    db: Session,
    raw_text: str,
    extracted_skills: List[str],
    filename: str = "api_recommend",
) -> Resume:
    """Return the latest resume with matching text or create a new one."""
    existing = (
        db.query(Resume)
        .filter(Resume.raw_text == raw_text)
        .order_by(Resume.created_at.desc())
        .first()
    )
    if existing:
        return existing

    return save_parsed_resume(db, filename, raw_text, extracted_skills)


def _find_source_job(
    ranked: Dict[str, Any], source_jobs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Match a ranked result back to the original job payload."""
    ranked_url = ranked.get("apply_url", "")
    ranked_title = ranked.get("title", "")
    ranked_company = ranked.get("company", "")

    for job in source_jobs:
        if ranked_url and job.get("apply_url") == ranked_url:
            return job
        if job.get("title") == ranked_title and job.get("company") == ranked_company:
            return job

    return {
        "title": ranked.get("title", ""),
        "company": ranked.get("company", ""),
        "location": ranked.get("location", ""),
        "description": ranked.get("description", ""),
        "requirements": ranked.get("requirements", ""),
        "apply_url": ranked.get("apply_url", ""),
        "job_type": ranked.get("job_type", "Unknown"),
        "source": ranked.get("source", "Unknown"),
    }


def save_recommendation_results(
    db: Session,
    resume_text: str,
    resume_skills: List[str],
    source_jobs: List[Dict[str, Any]],
    ranked_jobs: List[Dict[str, Any]],
) -> Resume:
    """Persist recommendation rows for a matching run."""
    resume = get_or_create_resume(db, resume_text, resume_skills)

    for ranked in ranked_jobs:
        job_data = _find_source_job(ranked, source_jobs)
        job = upsert_job(db, job_data)

        record = Recommendation(
            resume_id=resume.id,
            job_id=job.id,
            match_score=ranked.get("match_score", 0),
            matched_skills=ranked.get("matched_skills", []),
            missing_skills=ranked.get("missing_skills", []),
            reason=ranked.get("reason", ""),
        )
        db.add(record)

    db.commit()
    return resume


def get_recommendation_history(
    db: Session,
    limit: int = 50,
    resume_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return previous recommendations with resume and job details."""
    query = (
        db.query(Recommendation)
        .join(Resume)
        .join(Job)
        .order_by(Recommendation.created_at.desc())
    )

    if resume_id is not None:
        query = query.filter(Recommendation.resume_id == resume_id)

    rows = query.limit(limit).all()

    history: List[Dict[str, Any]] = []
    for row in rows:
        history.append(
            {
                "id": row.id,
                "resume_id": row.resume_id,
                "job_id": row.job_id,
                "match_score": row.match_score,
                "matched_skills": row.matched_skills or [],
                "missing_skills": row.missing_skills or [],
                "reason": row.reason,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "resume": {
                    "id": row.resume.id,
                    "filename": row.resume.filename,
                    "extracted_skills": row.resume.extracted_skills or [],
                },
                "job": {
                    "id": row.job.id,
                    "source": row.job.source,
                    "company": row.job.company,
                    "title": row.job.title,
                    "location": row.job.location,
                    "apply_url": row.job.apply_url,
                    "job_type": row.job.job_type,
                },
            }
        )

    return history


def _saved_job_to_dict(record: SavedJob) -> Dict[str, Any]:
    return {
        "id": record.id,
        "title": record.title,
        "company": record.company,
        "location": record.location,
        "apply_url": record.apply_url,
        "match_score": record.match_score,
        "missing_skills": record.missing_skills or [],
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


def save_bookmarked_job(db: Session, job_data: Dict[str, Any]) -> tuple[SavedJob, bool]:
    """Save a job bookmark. Returns (record, created_new)."""
    apply_url = job_data.get("apply_url", "").strip()
    if apply_url and apply_url != "#":
        existing = db.query(SavedJob).filter(SavedJob.apply_url == apply_url).first()
        if existing:
            return existing, False

    record = SavedJob(
        title=job_data.get("title", "Unknown Title"),
        company=job_data.get("company", "Unknown Company"),
        location=job_data.get("location", ""),
        apply_url=apply_url,
        match_score=int(job_data.get("match_score", 0) or 0),
        missing_skills=job_data.get("missing_skills", []) or [],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record, True


def list_saved_jobs(db: Session) -> List[Dict[str, Any]]:
    """Return all saved job bookmarks newest first."""
    rows = db.query(SavedJob).order_by(SavedJob.created_at.desc()).all()
    return [_saved_job_to_dict(row) for row in rows]


def delete_saved_job(db: Session, saved_job_id: int) -> bool:
    """Delete a saved job by id. Returns True if deleted."""
    record = db.query(SavedJob).filter(SavedJob.id == saved_job_id).first()
    if not record:
        return False
    db.delete(record)
    db.commit()
    return True
