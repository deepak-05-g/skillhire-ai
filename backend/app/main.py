from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import Base, engine
from app.models import Job, Recommendation, Resume, SavedJob  # noqa: F401 — register models
from app.routers import resume
from app.routers import jobs
from app.routers import recommend
from app.routers import chat
from app.routers import auth

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="SkillHire AI - AI-powered resume-based job recommendation system",
    version="0.1.0"
)

# Create database tables on startup
Base.metadata.create_all(bind=engine)

# Auto-train ML model on startup if missing
import os
from app.services.fit_classifier import MODEL_PATH, train_and_save_model

if not os.path.exists(MODEL_PATH):
    try:
        print("Model file not found. Auto-training classifier on startup...")
        train_and_save_model()
        print("Model trained and saved successfully.")
    except Exception as e:
        print(f"Error auto-training classifier model on startup: {e}")


# Configure CORS for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(resume.router, prefix=settings.API_STR)
app.include_router(jobs.router, prefix=settings.API_STR)
app.include_router(recommend.router, prefix=settings.API_STR)
app.include_router(chat.router, prefix=settings.API_STR)
app.include_router(auth.router)

@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint to verify backend status.
    """
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": "0.1.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
