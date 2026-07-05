from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR = os.path.dirname(_BACKEND_DIR)
_DEFAULT_DB_PATH = os.path.join(_BACKEND_DIR, "skillhire_ai.db")

class Settings(BaseSettings):
    PROJECT_NAME: str = "SkillHire AI"
    API_STR: str = "/api/v1"

    # Database Configuration
    DATABASE_URL: str = f"sqlite:///{_DEFAULT_DB_PATH}"

    # CORS Origins (comma-separated string, e.g. "http://localhost:8501,http://frontend:8501")
    ALLOWED_ORIGINS: str = "*"
    FRONTEND_URL: str = "http://localhost:8501"

    @property
    def allowed_origins(self) -> List[str]:
        if self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]

    # Job Board Integration API keys or base URLs if needed (placeholders)
    GREENHOUSE_API_URL: str = "https://boards-api.greenhouse.io/v1"
    LEVER_API_URL: str = "https://api.lever.co/v1"
    ASHBY_API_URL: str = "https://api.ashbyhq.com"

    # AI Configuration
    GEMINI_API_KEY: Optional[str] = None

    # Firebase Google Sign-In Configuration
    FIREBASE_API_KEY: Optional[str] = None
    FIREBASE_AUTH_DOMAIN: Optional[str] = None
    FIREBASE_PROJECT_ID: Optional[str] = None
    FIREBASE_STORAGE_BUCKET: Optional[str] = None
    FIREBASE_MESSAGING_SENDER_ID: Optional[str] = None
    FIREBASE_APP_ID: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=os.path.join(_PROJECT_DIR, ".env"),
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
