from dataguard.detection.regex import RegexPIIDetector
from dataguard.domain.models import PIIType


def test_detects_labeled_french_name_and_address() -> None:
    text = "Nom: Jean Tremblay\nAdresse: 1234 rue Sainte-Catherine, Montréal, Québec"
    detections = RegexPIIDetector().detect(text)
    values = {(item.pii_type, item.value) for item in detections}
    assert (PIIType.PERSON, "Jean Tremblay") in values
    assert any(item.pii_type is PIIType.ADDRESS for item in detections)


def test_detects_quebec_health_insurance_identifier() -> None:
    text = "NAM: ABCD 12345678"
    detections = RegexPIIDetector().detect(text)
    assert any(item.pii_type is PIIType.HEALTH_INSURANCE_ID for item in detections)


def test_detects_labeled_health_information() -> None:
    text = "Diagnostic: hypertension essentielle"
    detections = RegexPIIDetector().detect(text)
    assert any(item.pii_type is PIIType.HEALTH_INFORMATION for item in detections)
