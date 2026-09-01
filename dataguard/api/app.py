from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dataguard.api.dependencies import Principal, require_permission
from dataguard.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DetectionResponse,
    GovernanceResponse,
    PIARequest,
    PIAResponse,
    PIATransitionRequest,
    RemediationRequest,
    RemediationResponse,
    RiskResponse,
)
from dataguard.audit.service import AuditService
from dataguard.compliance.engine import ComplianceEngine
from dataguard.compliance.loader import FrameworkLoader
from dataguard.core.config import get_settings
from dataguard.database.models import Analysis, AuditEvent, PIARecord, RemediationItem
from dataguard.database.session import engine, get_session
from dataguard.detection.ensemble import EnsembleDetector
from dataguard.detection.pipeline import PIIDetectionPipeline
from dataguard.detection.regex import RegexPIIDetector
from dataguard.pia.models import PIA, PIAStatus
from dataguard.pia.workflow import PIAWorkflow
from dataguard.processing.models import DocumentInput
from dataguard.processing.pipeline import DocumentProcessingPipeline
from dataguard.risk.engine import RiskContext, RiskEngine
from dataguard.security.rate_limit_middleware import RateLimitMiddleware
from dataguard.security.request_limits import RequestSizeLimitMiddleware
from dataguard.security.web import SecurityHeadersMiddleware


def _redact(value: str | None) -> str:
    if not value:
        return "[REDACTED]"
    return "[REDACTED]" if len(value) <= 4 else "[REDACTED]" + value[-2:]


async def _tenant_session(session: AsyncSession, organization_id: str) -> None:
    try:
        UUID(organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid tenant") from exc
    await session.execute(
        text("SELECT set_config('dataguard.organization_id', :org, true)"),
        {"org": organization_id},
    )


def _governance(text_value: str, request: AnalyzeRequest) -> GovernanceResponse | None:
    if not request.framework:
        return None
    root = Path(__file__).resolve().parents[2] / "compliance" / "frameworks"
    try:
        rules = FrameworkLoader(root).load(request.framework)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown compliance framework: {request.framework}",
        ) from exc
    evidence = {
        "pii_detected": bool(text_value),
        "encryption_at_rest": request.encrypted_at_rest,
        "purpose_defined": request.purpose_defined,
        "retention_defined": request.retention_days is not None,
        "access_scope_defined": request.access_scope != "unknown",
        "data_location_defined": request.data_location != "unknown",
    }
    findings = ComplianceEngine(rules).evaluate(evidence)
    return GovernanceResponse(
        framework=request.framework,
        findings=[
            {
                "rule_id": f.rule_id,
                "framework": f.framework,
                "status": f.status,
                "severity": f.severity,
                "reason": f.reason,
                "required_evidence": list(f.required_evidence),
                "remediation": list(f.remediation),
            }
            for f in findings
        ],
    )


def _analysis_payload(detections, risk, governance):
    return {
        "detections": [
            {
                "type": d.pii_type.value,
                "start": d.start,
                "end": d.end,
                "confidence": d.confidence,
                "detector": d.detector,
                "redacted_value": _redact(d.value),
            }
            for d in detections
        ],
        "risk": {
            "score": risk.score,
            "level": risk.level.value,
            "factors": list(risk.factors),
            "explanation": risk.explanation,
            "recommendations": list(risk.recommendations),
        },
        "governance": governance.model_dump() if governance else None,
    }


async def _persist_analysis(
    session: AsyncSession,
    principal: Principal,
    source_type: str,
    source_ref: str | None,
    payload: dict,
) -> Analysis:
    analysis = Analysis(
        organization_id=UUID(principal.organization_id),
        source_type=source_type,
        source_ref=source_ref,
        status="COMPLETED",
        result=payload,
    )
    session.add(analysis)
    await session.flush()

    previous = await session.scalar(
        select(AuditEvent)
        .where(AuditEvent.organization_id == UUID(principal.organization_id))
        .order_by(AuditEvent.occurred_at.desc(), AuditEvent.id.desc())
        .limit(1)
    )
    previous_hash = previous.integrity_hash if previous else ""
    occurred_at = datetime.now(timezone.utc)
    actor_id = None
    try:
        actor_id = UUID(principal.subject)
    except ValueError:
        pass
    record = AuditService().create_record(
        event_id=str(analysis.id),
        timestamp=occurred_at.isoformat(),
        user_id=principal.subject,
        organization_id=principal.organization_id,
        action="ANALYSIS_COMPLETED",
        object_type="analysis",
        object_id=str(analysis.id),
        previous_state=None,
        new_state={"status": analysis.status, "source_type": source_type},
        ip_address=None,
        request_id=None,
        result="SUCCESS",
        previous_hash=previous_hash,
    )
    session.add(
        AuditEvent(
            organization_id=UUID(principal.organization_id),
            actor_id=actor_id,
            action=record.action,
            object_type=record.object_type,
            object_id=record.object_id,
            previous_state=record.previous_state,
            new_state=record.new_state,
            request_id=record.request_id,
            ip_address=record.ip_address,
            result=record.result,
            occurred_at=occurred_at,
            previous_hash=record.previous_hash,
            integrity_hash=record.integrity_hash,
        )
    )
    await session.commit()
    return analysis


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
        version="0.6.0",
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
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")
    pipeline = PIIDetectionPipeline(EnsembleDetector([RegexPIIDetector()]))
    risk_engine = RiskEngine()
    document_pipeline = DocumentProcessingPipeline()

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
        return {
            "status": "ok" if all(v == "ok" for v in checks.values()) else "degraded",
            **checks,
        }
