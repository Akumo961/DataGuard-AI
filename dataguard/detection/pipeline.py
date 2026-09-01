from __future__ import annotations

from dataguard.detection.base import DetectionEngine
from dataguard.detection.validation import DetectionValidator
from dataguard.domain.models import Detection


class PIIDetectionPipeline:
    def __init__(
        self, engine: DetectionEngine, validator: DetectionValidator | None = None
    ) -> None:
        self.engine = engine
        self.validator = validator or DetectionValidator()

    def detect(self, text: str) -> list[Detection]:
        return self.validator.validate(text, self.engine.detect(text))
