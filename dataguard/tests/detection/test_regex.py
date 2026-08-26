from dataguard.detection.regex import RegexPIIDetector
from dataguard.detection.pipeline import PIIDetectionPipeline
from dataguard.domain.models import PIIType


def test_detects_common_pii_types() -> None:
    text = "Contact alice@example.ca or +1 514-555-0199. Server 192.168.1.20. DOB 1985-04-23. SIN 046 454 286."
    detections = PIIDetectionPipeline(RegexPIIDetector()).detect(text)
    types = {item.pii_type for item in detections}
    assert PIIType.EMAIL in types
    assert PIIType.PHONE in types
    assert PIIType.IP_ADDRESS in types
    assert PIIType.DATE_OF_BIRTH in types
    assert PIIType.SOCIAL_INSURANCE_NUMBER in types


def test_credit_card_requires_luhn() -> None:
    detector = RegexPIIDetector()
    valid = detector.detect("4111 1111 1111 1111")
    invalid = detector.detect("4111 1111 1111 1112")
    assert any(item.pii_type is PIIType.CREDIT_CARD for item in valid)
    assert not any(item.pii_type is PIIType.CREDIT_CARD for item in invalid)


def test_empty_text_is_safe() -> None:
    assert RegexPIIDetector().detect("") == []
