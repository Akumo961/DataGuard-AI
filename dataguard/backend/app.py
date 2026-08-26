from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dataguard.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def readiness() -> dict[str, str]:
        # Dependency checks will be added with the persistence layer.
        return {"status": "ok"}

    return app


app = create_app()
