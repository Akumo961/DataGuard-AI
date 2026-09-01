from dataguard.detection.regex import RegexPIIDetector
from dataguard.domain.models import PIIType


def test_person_detected_without_field_label() -> None:
    detections = RegexPIIDetector().detect("Please contact Dr Jean Tremblay about the request.")
    assert any(
        d.pii_type is PIIType.PERSON and "Jean Tremblay" in d.value for d in detections
    )


def test_address_detected_without_field_label() -> None:
    text = "The office is at 123 Rue Sainte-Catherine, Montreal, QC H2X 1Z6."
    detections = RegexPIIDetector().detect(text)
    assert any(
        d.pii_type is PIIType.ADDRESS and "123 Rue Sainte-Catherine" in d.value
        for d in detections
    )


def test_contextual_rules_do_not_classify_arbitrary_words_as_person() -> None:
    detections = RegexPIIDetector().detect("The Finance Department reviewed the annual report.")
    assert not any(d.pii_type is PIIType.PERSON for d in detections)
