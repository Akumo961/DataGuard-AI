from __future__ import annotations

import jwt

from dataguard.security.auth import create_access_token, decode_access_token
from dataguard.security.policy import Role


def test_locally_issued_access_token_has_unique_jti(monkeypatch) -> None:
    monkeypatch.setenv("DATAGUARD_ENVIRONMENT", "development")
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", "test-secret-with-at-least-32-characters")
    first = create_access_token(
        subject_id="user-1", organization_id="org-1", roles={Role.ANALYST}
    )
    second = create_access_token(
        subject_id="user-1", organization_id="org-1", roles={Role.ANALYST}
    )
    first_claims = jwt.decode(first, options={"verify_signature": False})
    second_claims = jwt.decode(second, options={"verify_signature": False})
    assert first_claims["jti"]
    assert first_claims["jti"] != second_claims["jti"]
    assert decode_access_token(first).jti == first_claims["jti"]
