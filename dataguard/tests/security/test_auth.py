import jwt
import pytest

from dataguard.security.auth import create_access_token, decode_access_token
from dataguard.security.policy import Role


def test_local_access_token_round_trip(monkeypatch) -> None:
    monkeypatch.setenv("DATAGUARD_ENVIRONMENT", "development")
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", "x" * 48)
    from dataguard.core.config import get_settings
    get_settings.cache_clear()
    token = create_access_token(
        subject_id="user-1", organization_id="org-1", roles={Role.ANALYST}, expires_minutes=10
    )
    principal = decode_access_token(token)
    assert principal.subject_id == "user-1"
    assert principal.organization_id == "org-1"
    assert principal.roles == frozenset({Role.ANALYST})


def test_tampered_token_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("DATAGUARD_ENVIRONMENT", "development")
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", "x" * 48)
    from dataguard.core.config import get_settings
    get_settings.cache_clear()
    token = create_access_token(subject_id="u", organization_id="o", roles={Role.VIEWER})
    header, payload, signature = token.split(".")
    tampered = jwt.utils.base64url_encode(b'{"sub":"attacker"}').decode()
    with pytest.raises(jwt.InvalidTokenError):
        decode_access_token(f"{header}.{tampered}.{signature}")
