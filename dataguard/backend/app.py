"""FastAPI composition root and security boundary."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dataguard.core.config import get_settings
from dataguard.security.rate_limit_middleware import RateLimitMiddleware
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


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.3.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def readiness() -> dict[str, str]:
        if settings.environment == "test":
            return {"status": "ok"}
        redis: Redis | None = getattr(app.state, "redis", None)
        if redis is None:
            return {"status": "degraded"}
        try:
            await redis.ping()
        except Exception:
            return {"status": "degraded"}
        return {"status": "ok"}

    return app


app = create_app()
