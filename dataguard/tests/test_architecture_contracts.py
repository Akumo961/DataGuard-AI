from dataguard.application.analyze import AnalysisDependencies, AnalyzeText
from dataguard.compliance.base import ComplianceEngine, ControlFinding
from dataguard.database.ports import TenantContext
from dataguard.detection.base import DetectionEngine
from dataguard.domain.models import Detection, PIIType, RiskAssessment, RiskLevel
from dataguard.risk.base import RiskEngine


class StubDetector(DetectionEngine):
    name = "stub"

    def detect(self, text: str) -> list[Detection]:
        return [Detection(PIIType.EMAIL, 0, 4, 0.99, self.name)]


class StubRisk(RiskEngine):
    def assess(self, detections, context):
        return RiskAssessment(0.5, RiskLevel.MEDIUM, ({"factor": "test"},), "test", ("review",))


class StubCompliance(ComplianceEngine):
    def evaluate(self, context):
        return [ControlFinding("test", "TEST-1", "review", "medium", ("evidence",), ("review",))]


def test_domain_pipeline_is_composable():
    use_case = AnalyzeText(AnalysisDependencies((StubDetector(),), None, StubRisk(), StubCompliance()))
    result = use_case.execute(TenantContext("org-1", "user-1"), "test@example.com", {"purpose": "test"})
    assert result.analysis_id
    assert len(result.detections) == 1
    assert result.detections[0].detector == "stub"
    assert result.risk.level is RiskLevel.MEDIUM
    assert result.framework_findings[0]["rule_id"] == "TEST-1"


def test_empty_text_is_rejected():
    use_case = AnalyzeText(AnalysisDependencies((), None, StubRisk(), None))
    try:
        use_case.execute(TenantContext("org-1", None), " ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty input must be rejected")
