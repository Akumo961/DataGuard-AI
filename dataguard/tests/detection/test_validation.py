from dataguard.detection.validation import DetectionValidator
from dataguard.domain.models import Detection, PIIType


def test_validator_rejects_short_phone() -> None:
    detection = Detection(PIIType.PHONE, 0, 8, 0.9, "test", "123-4567")
    assert DetectionValidator().validate(detection.value or "", [detection]) == []


def test_validator_preserves_valid_email() -> None:
    detection = Detection(PIIType.EMAIL, 0, 13, 0.98, "test", "a@example.com")
    assert DetectionValidator().validate(detection.value or "", [detection]) == [detection]
