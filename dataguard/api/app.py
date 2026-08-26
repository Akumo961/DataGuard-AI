from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dataguard.api.dependencies import Principal, get_principal
from dataguard.api.schemas import AnalyzeRequest, AnalyzeResponse, DetectionResponse, RiskResponse
from dataguard.detection.ensemble import EnsembleDetector
from dataguard.detection.pipeline import PIIDetectionPipeline
from dataguard.detection.regex import RegexPIIDetector
from dataguard.risk.engine import RiskContext, RiskEngine


def _redact(value: str | None) -> str:
    if not value:
        return "[REDACTED]"
    return "[REDACTED]" if len(value) <= 4 else "[REDACTED]" + value[-2:]


def create_app() -> FastAPI:
    app = FastAPI(title="DataGuard Québec API", version="0.4.0", docs_url="/docs", redoc_url="/redoc")
    app.add_middleware(CORSMiddleware, allow_origins=[], allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"], max_age=600)

    pipeline = PIIDetectionPipeline(EnsembleDetector([RegexPIIDetector()]))
    risk_engine = RiskEngine()

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def readiness() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/api/v1/analyze", response_model=AnalyzeResponse)
    def analyze(request: AnalyzeRequest, principal: Principal = Depends(get_principal)) -> AnalyzeResponse:
        detections = pipeline.detect(request.text)
        risk = risk_engine.assess(detections, RiskContext(
            data_location=request.data_location,
            access_scope=request.access_scope,
            retention_days=request.retention_days,
            encrypted_at_rest=request.encrypted_at_rest,
            purpose_defined=request.purpose_defined,
            exposure=request.exposure,
            framework=request.framework,
        ))
        return AnalyzeResponse(
            organization_id=principal.organization_id,
            detections=[DetectionResponse(type=d.pii_type.value, start=d.start, end=d.end, confidence=d.confidence, detector=d.detector, redacted_value=_redact(d.value)) for d in detections],
            risk=RiskResponse(score=risk.score, level=risk.level.value, factors=list(risk.factors), explanation=risk.explanation, recommendations=list(risk.recommendations)),
        )

    return app


app = create_app()
