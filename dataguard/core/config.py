"""Shared configuration for API and workers.

Secrets are supplied by environment variables or a managed secret store;
no secret values belong in source control.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATAGUARD_", env_file=".env", extra="ignore")

    app_name: str = "DataGuard Québec"
    environment: str = Field(default="development", pattern=r"^(development|test|staging|production)$")
    api_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://dataguard:dataguard@localhost:5432/dataguard"
    redis_url: str = "redis://localhost:6379/0"
    allowed_origins: list[str] = ["http://localhost:3000"]
    max_upload_bytes: int = 50 * 1024 * 1024
    raw_document_retention_days: int = 7
    audit_retention_days: int = 2555
    request_timeout_seconds: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
