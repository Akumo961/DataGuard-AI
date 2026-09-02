from dataguard.classification.rules import RuleBasedClassifier
from dataguard.domain.models import Detection, PIIType


def detection(pii_type: PIIType, confidence: float = 0.9) -> Detection:
    return Detection(pii_type, 0, 4, confidence, "test", "redacted")


def test_credit_card_is_highly_restricted() -> None:
    result = RuleBasedClassifier().classify("", [detection(PIIType.CREDIT_CARD)], {})
    assert result.label == "HIGHLY_RESTRICTED"
    assert result.model_version == "rules-v1"


def test_email_is_confidential() -> None:
    result = RuleBasedClassifier().classify("", [detection(PIIType.EMAIL)], {})
    assert result.label == "CONFIDENTIAL"


def test_no_detection_is_public_but_explicitly_not_proof() -> None:
    result = RuleBasedClassifier().classify("", [], {})
    assert result.label == "PUBLIC"
    assert "not proof" in result.rationale
