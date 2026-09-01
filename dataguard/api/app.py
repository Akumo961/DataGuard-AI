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


def _governance(
    text_value: str, request: AnalyzeRequest
) -> GovernanceResponse | None:
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
    session.add(
        AuditEvent(
            organization_id=UUID(principal.organization_id),
            actor_id=None,
            action="ANALYSIS_COMPLETED",
            object_type="analysis",
            object_id=str(analysis.id),
            previous_state=None,
            new_state={"status": analysis.status, "source_type": source_type},
            request_id=None,
            ip_address=None,
            result="SUCCESS",
            occurred_at=datetime.now(timezone.utc),
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

    @app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["analysis"])
    async def analyze(
        request: AnalyzeRequest,
        principal: Principal = Depends(require_permission("analysis:write")),
        session: AsyncSession = Depends(get_session),
    ) -> AnalyzeResponse:
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
        governance = _governance(request.text, request)
        payload = _analysis_payload(detections, risk, governance)
        analysis_id = None
        if settings.environment != "test":
            await _tenant_session(session, principal.organization_id)
            analysis_id = str(
                (
                    await _persist_analysis(
                        session, principal, "text", None, payload
                    )
                ).id
            )
        return AnalyzeResponse(
            analysis_id=analysis_id,
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
            governance=governance,
        )

    @app.post(
        "/api/v1/analyze-document",
        response_model=AnalyzeResponse,
        tags=["analysis"],
    )
    async def analyze_document(
        file: UploadFile = File(...),
        principal: Principal = Depends(require_permission("analysis:write")),
        session: AsyncSession = Depends(get_session),
    ) -> AnalyzeResponse:
        content = await file.read()
        extracted = document_pipeline.process(
            DocumentInput(file.filename or "upload", content, file.content_type)
        )
        detections = pipeline.detect(extracted.text)
        risk = risk_engine.assess(detections, RiskContext())
        request = AnalyzeRequest(text=extracted.text)
        governance = _governance(extracted.text, request)
        payload = _analysis_payload(detections, risk, governance)
        payload["document"] = {
            "filename": extracted.filename,
            "document_type": extracted.document_type.value,
            "page_count": extracted.page_count,
            "warnings": list(extracted.warnings),
        }
        analysis_id = None
        if settings.environment != "test":
            await _tenant_session(session, principal.organization_id)
            analysis_id = str(
                (
                    await _persist_analysis(
                        session,
                        principal,
                        "document",
                        extracted.filename,
                        payload,
                    )
                ).id
            )
        return AnalyzeResponse(
            analysis_id=analysis_id,
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
            governance=governance,
        )

    @app.get("/api/v1/analyses/{analysis_id}", tags=["analysis"])
    async def get_analysis(
        analysis_id: str,
        principal: Principal = Depends(require_permission("analysis:read")),
        session: AsyncSession = Depends(get_session),
    ):
        try:
            aid = UUID(analysis_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Analysis not found") from exc
        await _tenant_session(session, principal.organization_id)
        row = (
            await session.execute(
                select(Analysis).where(
                    Analysis.id == aid,
                    Analysis.organization_id == UUID(principal.organization_id),
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return {
            "id": str(row.id),
            "organization_id": principal.organization_id,
            "status": row.status,
            "result": row.result,
        }

    @app.post("/api/v1/pias", response_model=PIAResponse, tags=["governance"])
    async def create_pia(
        request: PIARequest,
        principal: Principal = Depends(require_permission("pia:manage")),
        session: AsyncSession = Depends(get_session),
    ):
        await _tenant_session(session, principal.organization_id)
        row = PIARecord(
            organization_id=UUID(principal.organization_id),
            project_name=request.project_name,
            status="DRAFT",
            version=1,
            owner_id=principal.subject,
            payload=request.model_dump(),
        )
        session.add(row)
        await session.flush()
        session.add(
            AuditEvent(
                organization_id=UUID(principal.organization_id),
                actor_id=None,
                action="PIA_CREATED",
                object_type="pia",
                object_id=str(row.id),
                previous_state=None,
                new_state={"status": "DRAFT"},
                request_id=None,
                ip_address=None,
                result="SUCCESS",
                occurred_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
        return PIAResponse(
            id=str(row.id),
            organization_id=principal.organization_id,
            project_name=row.project_name,
            status=row.status,
            version=row.version,
        )

    @app.post(
        "/api/v1/pias/{pia_id}/transition",
        response_model=PIAResponse,
        tags=["governance"],
    )
    async def transition_pia(
        pia_id: str,
        request: PIATransitionRequest,
        principal: Principal = Depends(require_permission("pia:manage")),
        session: AsyncSession = Depends(get_session),
    ):
        await _tenant_session(session, principal.organization_id)
        try:
            pid = UUID(pia_id)
            target = PIAStatus(request.target)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid PIA id or status"
            ) from exc
        row = (
            await session.execute(
                select(PIARecord).where(
                    PIARecord.id == pid,
                    PIARecord.organization_id == UUID(principal.organization_id),
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="PIA not found")
        pia = PIA(
            pia_id=str(row.id),
            organization_id=principal.organization_id,
            project_name=row.project_name,
            owner_id=row.owner_id,
            status=PIAStatus(row.status),
            version=row.version,
            metadata=row.payload,
        )
        try:
            updated, entry = PIAWorkflow().transition(
                pia, target, principal.subject, request.reason
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        payload = dict(row.payload)
        history = list(payload.get("history", []))
        history.append(
            {
                "version": entry.version,
                "from_status": (
                    entry.from_status.value if entry.from_status else None
                ),
                "to_status": entry.to_status.value,
                "actor_id": entry.actor_id,
                "timestamp": entry.timestamp,
                "reason": entry.reason,
            }
        )
        payload["history"] = history
        row.status, row.version, row.payload = (
            updated.status.value,
            updated.version,
            payload,
        )
        session.add(
            AuditEvent(
                organization_id=UUID(principal.organization_id),
                actor_id=None,
                action="PIA_TRANSITIONED",
                object_type="pia",
                object_id=str(row.id),
                previous_state={"status": pia.status.value},
                new_state={"status": row.status, "version": row.version},
                request_id=None,
                ip_address=None,
                result="SUCCESS",
                occurred_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
        return PIAResponse(
            id=str(row.id),
            organization_id=principal.organization_id,
            project_name=row.project_name,
            status=row.status,
            version=row.version,
        )

    @app.post(
        "/api/v1/remediations",
        response_model=RemediationResponse,
        tags=["governance"],
    )
    async def create_remediation(
        request: RemediationRequest,
        principal: Principal = Depends(require_permission("analysis:write")),
        session: AsyncSession = Depends(get_session),
    ):
        await _tenant_session(session, principal.organization_id)
        analysis_id = None
        if request.analysis_id:
            try:
                analysis_id = UUID(request.analysis_id)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="Invalid analysis_id"
                ) from exc
            exists = (
                await session.execute(
                    select(Analysis.id).where(
                        Analysis.id == analysis_id,
                        Analysis.organization_id == UUID(principal.organization_id),
                    )
                )
            ).scalar_one_or_none()
            if exists is None:
                raise HTTPException(status_code=404, detail="Analysis not found")
        row = RemediationItem(
            organization_id=UUID(principal.organization_id),
            analysis_id=analysis_id,
            title=request.title,
            description=request.description,
            priority=request.priority.upper(),
            owner_id=request.owner_id or principal.subject,
            status="OPEN",
            evidence={},
        )
        session.add(row)
        await session.flush()
        session.add(
            AuditEvent(
                organization_id=UUID(principal.organization_id),
                actor_id=None,
                action="REMEDIATION_CREATED",
                object_type="remediation",
                object_id=str(row.id),
                previous_state=None,
                new_state={"status": "OPEN", "priority": row.priority},
                request_id=None,
                ip_address=None,
                result="SUCCESS",
                occurred_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
        return RemediationResponse(
            id=str(row.id),
            organization_id=principal.organization_id,
            status=row.status,
            priority=row.priority,
        )

    return app


app = create_app()
