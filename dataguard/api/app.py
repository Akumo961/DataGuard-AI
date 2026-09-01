from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis
from sqlalchemy import text
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dataguard.api.dependencies import Principal, get_principal
from dataguard.api.schemas import AnalyzeRequest, AnalyzeResponse, DetectionResponse, RiskResponse
from dataguard.core.config import get_settings
from dataguard.database.session import engine
from dataguard.detection.ensemble import EnsembleDetector
from dataguard.detection.pipeline import PIIDetectionPipeline
from dataguard.detection.regex import RegexPIIDetector
from dataguard.risk.engine import RiskContext, RiskEngine
from dataguard.security.rate_limit_middleware import RateLimitMiddleware
from dataguard.security.request_limits import RequestSizeLimitMiddleware
from dataguard.security.web import SecurityHeadersMiddleware


def _redact(value: str | None) -> str:
    if not value:
        return "[REDACTED]"
    return "[REDACTED]" if len(value) <= 4 else "[REDACTED]" + value[-2:]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    redis: Redis | None = None
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
        version="0.5.1",
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
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

    pipeline = PIIDetectionPipeline(EnsembleDetector([RegexPIIDetector()]))
    risk_engine = RiskEngine()

    @app.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

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

    @app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["analysis"])
    def analyze(request: AnalyzeRequest, principal: Principal = Depends(get_principal)) -> AnalyzeResponse:
        detections = pipeline.detect(request.text)
        risk = risk_engine.assess(
            detections,
            RiskContext(
                data_location=request.data_location,
                access_scope=request.access_scope,
                retention_days=request.retention_days,
                encrypted_at_rest=request.encrypted_at_rest,
                purpose_defined=request.purpose_defined,
                exposure=request.exposure,
                framework=request.framework,
            ),
        )
        return AnalyzeResponse(
            organization_id=principal.organization_id,
            detections=[
                DetectionResponse(
                    type=d.pii_type.value,
                    start=d.start,
                    end=d.end,
                    confidence=d.confidence,
                    detector=d.detector,
                    redacted_value=_redact(d.value),
                )
                for d in detections
            ],
            risk=RiskResponse(
                score=risk.score,
                level=risk.level.value,
                factors=list(risk.factors),
                explanation=risk.explanation,
                recommendations=list(risk.recommendations),
            ),
        )

    return app


app = create_app()
