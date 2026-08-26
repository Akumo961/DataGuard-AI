"""FastAPI composition root and security boundary."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from sqlalchemy import text
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dataguard.core.config import get_settings
from dataguard.database.session import engine
from dataguard.security.rate_limit_middleware import RateLimitMiddleware
from dataguard.security.request_limits import RequestSizeLimitMiddleware
from dataguard.security.web import SecurityHeadersMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    redis = None
    if settings.environment != "test":
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        app.state.redis = redis
    try:
        yield
    finally:
        if redis is not None:
            await redis.aclose()
        await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.4.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                       allow_headers=["Authorization", "Content-Type", "X-Request-ID"], max_age=600)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def readiness() -> dict[str, str]:
        if settings.environment == "test":
            return {"status": "ok"}
        checks: dict[str, str] = {}
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception:
            checks["database"] = "degraded"
        redis: Redis | None = getattr(app.state, "redis", None)
        if redis is not None:
            try:
                await redis.ping()
                checks["redis"] = "ok"
            except Exception:
                checks["redis"] = "degraded"
        else:
            checks["redis"] = "disabled"
        status = "ok" if all(value == "ok" for value in checks.values()) else "degraded"
        return {"status": status, **checks}

    return app


app = create_app()
