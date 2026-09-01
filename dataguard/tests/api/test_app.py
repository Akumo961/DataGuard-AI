from datetime import datetime, timedelta, timezone

import jwt
from fastapi.testclient import TestClient

from dataguard.api.app import create_app
from dataguard.core.config import get_settings

TEST_SECRET = "test-secret-with-at-least-32-characters"


def token(org_id: str = "org-1", roles: list[str] | None = None) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": "user-1",
            "org_id": org_id,
            "roles": roles or ["analyst"],
            "iat": now,
            "exp": now + timedelta(minutes=5),
        },
        TEST_SECRET,
        algorithm="HS256",
    )


def test_health_does_not_require_auth(monkeypatch) -> None:
    monkeypatch.setenv("DATAGUARD_ENVIRONMENT", "test")
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", TEST_SECRET)
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        assert client.get("/health/live").status_code == 200
    finally:
        get_settings.cache_clear()


def test_analyze_requires_auth(monkeypatch) -> None:
    monkeypatch.setenv("DATAGUARD_ENVIRONMENT", "test")
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", TEST_SECRET)
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        response = client.post("/api/v1/analyze", json={"text": "Email alice@example.com"})
        assert response.status_code == 401
    finally:
        get_settings.cache_clear()


def test_analyze_is_tenant_scoped_and_redacts_value(monkeypatch) -> None:
    monkeypatch.setenv("DATAGUARD_ENVIRONMENT", "test")
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", TEST_SECRET)
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/v1/analyze",
            headers={"Authorization": f"Bearer {token('org-42')}"},
            json={"text": "Email alice@example.com"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["organization_id"] == "org-42"
        assert body["detections"][0]["redacted_value"].startswith("[REDACTED]")
        assert "alice@example.com" not in response.text
    finally:
        get_settings.cache_clear()


def test_viewer_cannot_write_analysis(monkeypatch) -> None:
    monkeypatch.setenv("DATAGUARD_ENVIRONMENT", "test")
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", TEST_SECRET)
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/v1/analyze",
            headers={"Authorization": f"Bearer {token('org-42', ['viewer'])}"},
            json={"text": "Email alice@example.com"},
        )
        assert response.status_code == 403
    finally:
        get_settings.cache_clear()


def test_analyst_cannot_manage_pia(monkeypatch) -> None:
    monkeypatch.setenv("DATAGUARD_ENVIRONMENT", "test")
    monkeypatch.setenv("DATAGUARD_JWT_SECRET", TEST_SECRET)
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/v1/pias",
            headers={"Authorization": f"Bearer {token('org-42', ['analyst'])}"},
            json={"project_name": "Restricted"},
        )
        assert response.status_code == 403
    finally:
        get_settings.cache_clear()
