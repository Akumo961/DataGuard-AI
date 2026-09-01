from types import SimpleNamespace

from dataguard.detection.ner import NERDetector
from dataguard.domain.models import PIIType


class FakeModel:
    def __call__(self, text: str):
        start = text.index("Jean Tremblay")
        return [
            SimpleNamespace(
                label_="PER",
                start_char=start,
                end_char=start + len("Jean Tremblay"),
                score=0.91,
            )
        ]


def test_ner_adapter_maps_person_label_and_preserves_span() -> None:
    detector = NERDetector(FakeModel(), {"PER": PIIType.PERSON})
    result = detector.detect("Jean Tremblay travaille à Montréal.")
    assert len(result) == 1
    assert result[0].pii_type is PIIType.PERSON
    assert result[0].value == "Jean Tremblay"
    assert result[0].confidence == 0.91
