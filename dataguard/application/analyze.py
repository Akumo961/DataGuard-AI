from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from dataguard.classification.base import Classifier
from dataguard.compliance.base import ComplianceEngine
from dataguard.database.ports import AnalysisRepository, TenantContext
from dataguard.detection.base import DetectionEngine
from dataguard.domain.models import AnalysisResult
from dataguard.risk.base import RiskEngine


@dataclass(frozen=True)
class AnalysisDependencies:
    detectors: tuple[DetectionEngine, ...]
    classifier: Classifier | None
    risk_engine: RiskEngine
    compliance_engine: ComplianceEngine | None
    repository: AnalysisRepository | None = None


class AnalyzeText:
    """Application use case coordinating replaceable domain components."""

    def __init__(self, dependencies: AnalysisDependencies) -> None:
        self.dependencies = dependencies

    def execute(self, tenant: TenantContext, text: str, context: dict[str, Any] | None = None) -> AnalysisResult:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        ctx = dict(context or {})
        detections = [d for detector in self.dependencies.detectors for d in detector.detect(text)]
        risk = self.dependencies.risk_engine.assess(detections, ctx)
        findings = ()
        if self.dependencies.compliance_engine is not None:
            findings = tuple(vars(f) for f in self.dependencies.compliance_engine.evaluate(ctx))
        result = AnalysisResult(str(uuid4()), tuple(detections), risk, findings)
        if self.dependencies.repository is not None:
            self.dependencies.repository.save_analysis(tenant, {"analysis_id": result.analysis_id, "result": result})
        return result
