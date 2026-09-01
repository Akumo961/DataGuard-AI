from __future__ import annotations

import pytest

from dataguard.core.config import Settings


def _production_settings(**overrides):
    values = {
        "environment": "production",
        "jwt_algorithm": "RS256",
        "oidc_issuer_url": "https://issuer.example.test",
        "oidc_jwks_url": "https://issuer.example.test/.well-known/jwks.json",
        "jwt_issuer": "https://issuer.example.test",
        "jwt_audience": "dataguard",
        "allowed_origins": ["https://app.example.test"],
        "allowed_hosts": ["app.example.test", "api.example.test"],
        "database_url": "postgresql+psycopg://user:pass@db.internal:5432/dataguard",
        "redis_url": "redis://redis.internal:6379/0",
    }
    values.update(overrides)
    return Settings(**values)


def test_production_configuration_accepts_secure_external_dependencies() -> None:
    _production_settings().validate_production()


@pytest.mark.parametrize(
    "field,value",
    [
        ("allowed_hosts", ["localhost"]),
        ("database_url", "postgresql+psycopg://user:pass@localhost:5432/dataguard"),
        ("redis_url", "redis://localhost:6379/0"),
        ("security_headers_enabled", False),
    ],
)
def test_production_configuration_rejects_insecure_defaults(field: str, value) -> None:
    with pytest.raises(ValueError):
        _production_settings(**{field: value}).validate_production()


def test_production_configuration_requires_request_limit_to_cover_uploads() -> None:
    with pytest.raises(ValueError):
        _production_settings(
            max_upload_bytes=50 * 1024 * 1024,
            max_request_body_bytes=10 * 1024 * 1024,
        ).validate_production()
