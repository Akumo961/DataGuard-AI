from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATAGUARD_", env_file=".env", extra="ignore", case_sensitive=False
    )

    app_name: str = "DataGuard"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["http://localhost:3000"]
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "testserver"]
    database_url: str = "postgresql+psycopg://dataguard:dataguard@localhost:5432/dataguard"
    redis_url: str = "redis://localhost:6379/0"
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1_048_576, le=100 * 1024 * 1024)
    raw_document_retention_days: int = Field(default=7, ge=1, le=3650)
    audit_retention_days: int = Field(default=2555, ge=30, le=36500)
    request_timeout_seconds: int = Field(default=60, ge=1, le=300)
    jwt_secret: SecretStr | None = None
    jwt_algorithm: str = Field(default="HS256", pattern=r"^(HS256|RS256|ES256)$")
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    oidc_issuer_url: str | None = None
    oidc_jwks_url: str | None = None
    security_headers_enabled: bool = True
    rate_limit_per_minute: int = Field(default=120, ge=1, le=10_000)
    max_request_body_bytes: int = Field(default=60 * 1024 * 1024, ge=1_024, le=110 * 1024 * 1024)
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

    @field_validator("allowed_origins", "allowed_hosts")
    @classmethod
    def validate_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one origin/host must be configured")
        if any(item.strip() == "*" for item in value):
            raise ValueError("Wildcard CORS/trusted-host entries are forbidden")
        return value

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        return value

    @field_validator("oidc_issuer_url", "oidc_jwks_url")
    @classmethod
    def validate_oidc_urls(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("https://"):
            raise ValueError("OIDC endpoints must use HTTPS")
        return value

    def validate_production(self) -> Settings:
        if self.environment.lower() not in {"production", "prod"}:
            return self
        if self.jwt_algorithm not in {"RS256", "ES256"}:
            raise ValueError("Production authentication requires RS256 or ES256")
        if (
            not self.oidc_issuer_url
            or not self.oidc_jwks_url
            or not self.jwt_issuer
            or not self.jwt_audience
        ):
            raise ValueError(
                "Production authentication requires OIDC issuer, JWKS URL, JWT issuer and audience"
            )
        if any(origin.startswith("http://") for origin in self.allowed_origins):
            raise ValueError("Production CORS origins must use HTTPS")
        if any(
            host in {"localhost", "127.0.0.1", "[::1]", "testserver"}
            for host in self.allowed_hosts
        ):
            raise ValueError("Production trusted hosts must not contain local/test hosts")
        if any(host in self.database_url.lower() for host in ("localhost", "127.0.0.1")):
            raise ValueError("Production database must not use a local loopback host")
        if any(host in self.redis_url.lower() for host in ("localhost", "127.0.0.1")):
            raise ValueError("Production Redis must not use a local loopback host")
        if self.max_request_body_bytes < self.max_upload_bytes:
            raise ValueError("max_request_body_bytes must be >= max_upload_bytes")
        if not self.security_headers_enabled:
            raise ValueError("Production security headers cannot be disabled")
        return self


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_production()
    return settings
