import os

import jwt
import pytest
from fastapi import HTTPException

from dataguard.api.dependencies import get_principal


def test_missing_bearer_token_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", "test-secret")
    with pytest.raises(HTTPException) as exc:
        get_principal(None)
    assert exc.value.status_code == 401


def test_valid_token_produces_tenant_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test-secret"
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", secret)
    token = jwt.encode({"sub": "user-1", "org_id": "org-1", "roles": ["analyst"], "exp": 4102444800}, secret, algorithm="HS256")
    principal = get_principal(f"Bearer {token}")
    assert principal.subject == "user-1"
    assert principal.organization_id == "org-1"
    assert principal.roles == ("analyst",)
