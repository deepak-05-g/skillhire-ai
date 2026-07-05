from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.database import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(128), nullable=False, default="Unknown")
    company = Column(String(256), nullable=False)
    title = Column(String(512), nullable=False)
    location = Column(String(256), nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    requirements = Column(Text, nullable=False, default="")
    apply_url = Column(String(1024), nullable=False, default="")
    job_type = Column(String(64), nullable=False, default="Unknown")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
