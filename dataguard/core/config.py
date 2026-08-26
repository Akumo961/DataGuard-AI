"""Shared configuration for API and workers.

Secrets are supplied by environment variables or a managed secret store;
no secret values belong in source control.
"""

from functools import lru_cache
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATAGUARD_", env_file=".env", extra="ignore", case_sensitive=False
    )

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

    jwt_secret: SecretStr | None = None
    jwt_algorithm: str = Field(default="HS256", pattern=r"^(HS256|RS256|ES256)$")
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    oidc_issuer_url: str | None = None
    security_headers_enabled: bool = True
    rate_limit_per_minute: int = Field(default=120, ge=1, le=10_000)
    max_request_body_bytes: int = Field(default=60 * 1024 * 1024, ge=1_024)
    upload_allowed_mime_types: list[str] = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/csv",
        "application/json",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "image/png",
        "image/jpeg",
        "image/tiff",
    ]

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, value: list[str]) -> list[str]:
        if not value or any(origin.strip() == "*" for origin in value):
            raise ValueError("Wildcard CORS origins are forbidden")
        return [origin.rstrip("/") for origin in value]

    @field_validator("jwt_secret")
    @classmethod
    def validate_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("JWT secret must contain at least 32 characters")
        return value

    def validate_production(self) -> None:
        if self.environment != "production":
            return
        if self.jwt_algorithm == "HS256" and self.jwt_secret is None:
            raise ValueError("Production requires an explicit JWT validation secret or asymmetric OIDC setup")
        if self.jwt_algorithm in {"RS256", "ES256"}:
            raise ValueError("Configure an OIDC/JWKS validation adapter before enabling asymmetric JWTs")
        if self.oidc_issuer_url is None or self.jwt_issuer is None or self.jwt_audience is None:
            raise ValueError("Production authentication requires issuer, audience and OIDC configuration")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_production()
    return settings
