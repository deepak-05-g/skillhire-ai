from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, JSON, String

from app.database import Base


class SavedJob(Base):
    """Bookmarked job saved by the user (no login — local SQLite storage)."""

    __tablename__ = "saved_jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(512), nullable=False)
    company = Column(String(256), nullable=False)
    location = Column(String(256), nullable=False, default="")
    apply_url = Column(String(1024), nullable=False, default="")
    match_score = Column(Integer, nullable=False, default=0)
    missing_skills = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
