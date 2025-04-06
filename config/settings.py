import os
from typing import List, Optional

from pydantic import PostgresDsn, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration settings
    """

    PROJECT_NAME: str = "Medical Summarization API"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "development-secret-key")

    DATABASE_URL: Optional[PostgresDsn] = os.getenv("DATABASE_URL")

    ABSTRACTIVE_MODEL: str = "t5-small"
    MAX_SUMMARY_LENGTH: int = 150
    MIN_SUMMARY_LENGTH: int = 50

    CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://yourdomain.com"]
    ALLOWED_HOSTS: List[str] = ["localhost", "*.yourdomain.com"]

    LOG_LEVEL: str = "INFO"

    REQUEST_LIMIT_PER_MINUTE: int = 100

    @validator("DATABASE_URL", pre=True)
    def validate_database_url(cls, v):
        """
        Validate and transform database URL
        """
        if not v:
            return None
        return v

    class Config:
        """
        Pydantic configuration for environment variable loading
        """

        env_file = ".env"
        case_sensitive = True


settings = Settings()
