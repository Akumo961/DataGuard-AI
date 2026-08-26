from pathlib import Path

from fastapi.testclient import TestClient

from dataguard.api.app import create_app


def test_frontend_is_served() -> None:
    response = TestClient(create_app()).get("/")
    assert response.status_code == 200
    assert "DataGuard Québec" in response.text
    assert "Bearer token" in response.text


def test_frontend_assets_are_served() -> None:
    client = TestClient(create_app())
    assert client.get("/frontend/styles.css").status_code == 200
    assert client.get("/frontend/app.js").status_code == 200
