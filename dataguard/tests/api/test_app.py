from datetime import datetime, timedelta, timezone

import jwt
from fastapi.testclient import TestClient

from dataguard.api.app import create_app


def token(org_id: str = "org-1") -> str:
    return jwt.encode({"sub": "user-1", "org_id": org_id, "roles": ["analyst"], "exp": datetime.now(timezone.utc) + timedelta(minutes=5)}, "test-secret", algorithm="HS256")


def test_health_does_not_require_auth(monkeypatch) -> None:
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", "test-secret")
    client = TestClient(create_app())
    assert client.get("/health/live").status_code == 200


def test_analyze_requires_auth(monkeypatch) -> None:
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", "test-secret")
    client = TestClient(create_app())
    response = client.post("/api/v1/analyze", json={"text": "Email alice@example.com"})
    assert response.status_code == 401


def test_analyze_is_tenant_scoped_and_redacts_value(monkeypatch) -> None:
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", "test-secret")
    client = TestClient(create_app())
    response = client.post("/api/v1/analyze", headers={"Authorization": f"Bearer {token('org-42')}"}, json={"text": "Email alice@example.com"})
    assert response.status_code == 200
    body = response.json()
    assert body["organization_id"] == "org-42"
    assert body["detections"][0]["redacted_value"].startswith("[REDACTED]")
    assert "alice@example.com" not in response.text
