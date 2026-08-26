from dataguard.domain.models import Detection, PIIType, RiskLevel
from dataguard.risk.engine import RiskContext, RiskEngine


def test_empty_detection_is_low_but_not_proof_of_absence() -> None:
    result = RiskEngine().assess([])
    assert result.level is RiskLevel.LOW
    assert "not proof" in result.explanation


def test_sensitive_data_and_missing_controls_raise_risk() -> None:
    detections = [Detection(PIIType.SOCIAL_INSURANCE_NUMBER, 0, 11, 0.95, "regex", "046454286")]
    result = RiskEngine().assess(detections, RiskContext(access_scope="public", exposure="internet"))
    assert result.score >= 80
    assert result.level is RiskLevel.CRITICAL
    assert result.factors
    assert result.recommendations


def test_controls_reduce_risk_relative_to_missing_controls() -> None:
    detections = [Detection(PIIType.EMAIL, 0, 13, 0.9, "regex", "a@example.com")]
    insecure = RiskEngine().assess(detections)
    controlled = RiskEngine().assess(detections, RiskContext(encrypted_at_rest=True, purpose_defined=True, retention_days=30, data_location="canada", access_scope="internal", exposure="internal"))
    assert controlled.score < insecure.score
