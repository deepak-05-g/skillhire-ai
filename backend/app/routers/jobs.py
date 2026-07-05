from fastapi import APIRouter, Query, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.job_sources import fetch_jobs
from app.services.job_sources.official_search import generate_official_search_sources
from app.services.storage import (
    delete_saved_job,
    list_jobs,
    list_saved_jobs,
    save_bookmarked_job,
    save_jobs_batch,
)

router = APIRouter(
    prefix="/jobs",
    tags=["Jobs"]
)


class OfficialSearchRequest(BaseModel):
    resume_text: str = Field(..., description="Parsed resume text")
    resume_skills: List[str] = Field(default_factory=list, description="Extracted skills")
    location: Optional[str] = Field(default=None, description="Preferred job location")
    include_amazon: bool = Field(default=True, description="Include Amazon Jobs search link")


class OfficialSearchResponse(BaseModel):
    official_search_sources: List[dict]


class SaveJobRequest(BaseModel):
    title: str
    company: str
    location: str = ""
    apply_url: str
    match_score: int = 0
    missing_skills: List[str] = Field(default_factory=list)


class SavedJobItem(BaseModel):
    id: int
    title: str
    company: str
    location: str
    apply_url: str
    match_score: int
    missing_skills: List[str]
    created_at: Optional[str] = None


class SavedJobsResponse(BaseModel):
    saved_jobs: List[SavedJobItem]


class SaveJobResponse(BaseModel):
    saved_job: SavedJobItem
    message: str


class StoredJobsResponse(BaseModel):
    jobs: List[Dict[str, Any]]
    count: int


@router.get("/stored", response_model=StoredJobsResponse, status_code=status.HTTP_200_OK)
def list_stored_jobs_endpoint(
    db: Session = Depends(get_db),
    limit: int = Query(500, ge=1, le=1000, description="Max stored jobs to return"),
):
    """List jobs already stored in the database for resume matching."""
    jobs = list_jobs(db, limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@router.get("/fetch", response_model=List[Dict[str, Any]])
def fetch_jobs_endpoint(
    source: str = Query(..., description="Job source board (greenhouse, lever, or ashby)"),
    company: str = Query(..., description="Company board token or slug handle"),
    db: Session = Depends(get_db),
):
    """
    Fetches job listings from public company job boards and returns them in a normalized format.
    """
    source_clean = source.lower().strip()
    company_clean = company.strip()
    
    if not company_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Company parameter cannot be empty."
        )
        
    if source_clean not in ["greenhouse", "lever", "ashby"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid source. Supported job boards are 'greenhouse', 'lever', or 'ashby'."
        )
        
    try:
        jobs = fetch_jobs(source_clean, company_clean)
        if jobs:
            save_jobs_batch(db, jobs)
        return jobs
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during job fetching: {str(e)}"
        )


@router.post(
    "/official-search",
    response_model=OfficialSearchResponse,
    status_code=status.HTTP_200_OK,
)
def official_search_endpoint(request: OfficialSearchRequest):
    """
    Generate official Google, Microsoft, and Amazon career search URLs
    based on resume role/skills/location. No scraping — links only.
    """
    if not request.resume_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="resume_text cannot be empty.",
        )

    sources = generate_official_search_sources(
        resume_text=request.resume_text,
        resume_skills=request.resume_skills,
        location=request.location,
        include_amazon=request.include_amazon,
    )
    return {"official_search_sources": sources}


@router.post("/save", response_model=SaveJobResponse, status_code=status.HTTP_201_CREATED)
def save_job_endpoint(request: SaveJobRequest, db: Session = Depends(get_db)):
    """Bookmark a recommended job to SQLite (no login required)."""
    if not request.title.strip() or not request.company.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="title and company are required.",
        )

    record, created = save_bookmarked_job(db, request.model_dump())
    message = "Job saved successfully." if created else "Job was already saved."
    return {
        "saved_job": {
            "id": record.id,
            "title": record.title,
            "company": record.company,
            "location": record.location,
            "apply_url": record.apply_url,
            "match_score": record.match_score,
            "missing_skills": record.missing_skills or [],
            "created_at": record.created_at.isoformat() if record.created_at else None,
        },
        "message": message,
    }


@router.get("/saved", response_model=SavedJobsResponse, status_code=status.HTTP_200_OK)
def list_saved_jobs_endpoint(db: Session = Depends(get_db)):
    """List all bookmarked jobs."""
    return {"saved_jobs": list_saved_jobs(db)}


@router.delete("/saved/{saved_job_id}", status_code=status.HTTP_200_OK)
def delete_saved_job_endpoint(saved_job_id: int, db: Session = Depends(get_db)):
    """Remove a bookmarked job by id."""
    deleted = delete_saved_job(db, saved_job_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Saved job with id {saved_job_id} not found.",
        )
    return {"message": "Saved job removed.", "id": saved_job_id}
