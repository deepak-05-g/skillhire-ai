from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.matcher import rank_jobs
from app.services.storage import (
    get_recommendation_history,
    list_jobs,
    save_recommendation_results,
)

router = APIRouter(
    prefix="/recommend",
    tags=["Recommendations"],
)

# Pydantic schemas for request validation
class JobItem(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Software Engineer Intern",
                "company": "Google",
                "location": "Bengaluru, India",
                "description": "Looking for a software engineering intern...",
                "requirements": "Proficient in Python and SQL.",
                "apply_url": "https://careers.google.com",
                "job_type": "Internship",
                "source": "Greenhouse",
            }
        }
    )

    title: str
    company: str
    location: str
    description: str
    requirements: str
    apply_url: str
    job_type: str = "Unknown"
    source: str = "Unknown"


class RecommendRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "resume_text": "Software engineer with Python and SQL experience.",
                "resume_skills": ["Python", "SQL"],
                "use_ml": True,
            }
        }
    )

    resume_text: str
    resume_skills: List[str]
    jobs: List[JobItem] = Field(..., description="List of normalized jobs to rank")
    use_ml: bool = Field(default=True, description="Use ML model or heuristic fallback")


class RecommendStoredJobsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "resume_text": "Software engineer with Python and SQL experience.",
                "resume_skills": ["Python", "SQL"],
                "use_ml": True,
                "job_limit": 500,
            }
        }
    )

    resume_text: str
    resume_skills: List[str]
    use_ml: bool = Field(default=True, description="Use ML model or heuristic fallback")
    job_limit: int = Field(default=500, ge=1, le=1000, description="Stored jobs to rank")


class RecommendResponse(BaseModel):
    recommendations: List[dict]


class RecommendStoredJobsResponse(BaseModel):
    recommendations: List[dict]
    jobs_analyzed: int


class HistoryResponse(BaseModel):
    history: List[dict]
    count: int


@router.post("/jobs", response_model=RecommendResponse, status_code=status.HTTP_200_OK)
def recommend_jobs_endpoint(
    request: RecommendRequest,
    db: Session = Depends(get_db),
):
    """
    Ranks job opportunities against a candidate's resume based on semantic similarity,
    skill match percentage, and job title keywords.
    """
    if not request.jobs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The jobs list cannot be empty.",
        )

    try:
        jobs_dicts = [job.model_dump() for job in request.jobs]

        ranked_jobs = rank_jobs(
            resume_text=request.resume_text,
            resume_skills=request.resume_skills,
            jobs=jobs_dicts,
            use_ml=request.use_ml,
        )

        save_recommendation_results(
            db=db,
            resume_text=request.resume_text,
            resume_skills=request.resume_skills,
            source_jobs=jobs_dicts,
            ranked_jobs=ranked_jobs,
        )

        return {"recommendations": ranked_jobs}


    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while ranking jobs: {str(e)}",
        )


@router.post(
    "/stored-jobs",
    response_model=RecommendStoredJobsResponse,
    status_code=status.HTTP_200_OK,
)
def recommend_stored_jobs_endpoint(
    request: RecommendStoredJobsRequest,
    db: Session = Depends(get_db),
):
    """
    Rank jobs already stored in the database against a candidate resume.
    """
    if not request.resume_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="resume_text cannot be empty.",
        )

    jobs_dicts = list_jobs(db, limit=request.job_limit)
    if not jobs_dicts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No stored jobs found. Fetch jobs or load sample data first.",
        )

    try:
        ranked_jobs = rank_jobs(
            resume_text=request.resume_text,
            resume_skills=request.resume_skills,
            jobs=jobs_dicts,
            use_ml=request.use_ml,
        )

        save_recommendation_results(
            db=db,
            resume_text=request.resume_text,
            resume_skills=request.resume_skills,
            source_jobs=jobs_dicts,
            ranked_jobs=ranked_jobs,
        )

        return {
            "recommendations": ranked_jobs,
            "jobs_analyzed": len(jobs_dicts),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while ranking stored jobs: {str(e)}",
        )


@router.get("/history", response_model=HistoryResponse, status_code=status.HTTP_200_OK)
def recommend_history_endpoint(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200, description="Max records to return"),
    resume_id: Optional[int] = Query(None, description="Filter by resume ID"),
):
    """Return previously saved job recommendations from the database."""
    history = get_recommendation_history(db, limit=limit, resume_id=resume_id)
    return {"history": history, "count": len(history)}


@router.get("/ml-model/status", status_code=status.HTTP_200_OK)
def get_ml_model_status():
    """
    Get current machine learning classifier metrics, status, features, and feature importances.
    """
    import os
    from app.services.fit_classifier import load_model, MODEL_PATH
    
    if not os.path.exists(MODEL_PATH):
        return {
            "model_loaded": False,
            "metrics": None,
            "feature_names": [],
            "feature_importances": {}
        }
        
    try:
        bundle = load_model()
        if not bundle:
            return {
                "model_loaded": False,
                "metrics": None,
                "feature_names": [],
                "feature_importances": {}
            }
            
        model = bundle["model"]
        feature_names = bundle["feature_names"]
        metrics = bundle["metrics"]
        
        # Calculate feature importances from RandomForest
        importances = model.feature_importances_.tolist()
        feature_importances = dict(zip(feature_names, importances))
        
        return {
            "model_loaded": True,
            "metrics": metrics,
            "feature_names": feature_names,
            "feature_importances": feature_importances
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load ML model status: {str(e)}"
        )


@router.post("/ml-model/train", status_code=status.HTTP_200_OK)
def train_ml_model_endpoint():
    """
    Retrain the ML classifier on synthetic data and update the model file.
    """
    from app.services.fit_classifier import train_and_save_model
    try:
        metrics = train_and_save_model()
        return {
            "message": "Model retrained and saved successfully.",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model retraining failed: {str(e)}"
        )


@router.get("/ml-model/download", status_code=status.HTTP_200_OK)
def download_ml_model_endpoint():
    """
    Download the trained model pkl file.
    """
    import os
    from fastapi.responses import FileResponse
    from app.services.fit_classifier import MODEL_PATH
    
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model file not found. Please train the model first."
        )
        
    return FileResponse(
        path=MODEL_PATH,
        filename="fit_classifier.pkl",
        media_type="application/octet-stream"
    )
