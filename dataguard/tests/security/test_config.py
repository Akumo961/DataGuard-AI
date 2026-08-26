import pytest
from pydantic import ValidationError

from dataguard.core.config import Settings


def test_wildcard_cors_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(allowed_origins=["*"])


def test_production_rejects_hmac_jwt() -> None:
    settings = Settings(
        environment="production",
        jwt_secret="x" * 48,
        jwt_algorithm="HS256",
        oidc_issuer_url="https://id.example.gov",
        jwt_issuer="https://id.example.gov",
        jwt_audience="dataguard",
        allowed_origins=["https://dataguard.example.gov"],
    )
    with pytest.raises(ValueError, match="RS256 or ES256"):
        settings.validate_production()
