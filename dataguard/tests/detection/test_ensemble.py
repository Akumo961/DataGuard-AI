from dataguard.detection.base import DetectionEngine
from dataguard.detection.ensemble import EnsembleDetector
from dataguard.domain.models import Detection, PIIType


class StubDetector(DetectionEngine):
    name = "stub"

    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

    def detect(self, text: str) -> list[Detection]:
        return [Detection(PIIType.EMAIL, 0, 17, self.confidence, self.name, "a@example.com")]


def test_ensemble_combines_confidence() -> None:
    result = EnsembleDetector([StubDetector(0.7), StubDetector(0.6)]).detect("a@example.com")
    assert len(result) == 1
    assert result[0].confidence > 0.7
    assert result[0].detector == "ensemble"
