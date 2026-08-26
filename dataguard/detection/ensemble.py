from __future__ import annotations

from collections import defaultdict

from dataguard.detection.base import DetectionEngine
from dataguard.domain.models import Detection


class EnsembleDetector(DetectionEngine):
    name = "ensemble"

    def __init__(self, detectors: list[DetectionEngine], threshold: float = 0.50) -> None:
        if not detectors:
            raise ValueError("at least one detector is required")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.detectors = detectors
        self.threshold = threshold

    def detect(self, text: str) -> list[Detection]:
        candidates: list[Detection] = []
        for detector in self.detectors:
            candidates.extend(detector.detect(text))
        groups: dict[tuple[str, int, int], list[Detection]] = defaultdict(list)
        for item in candidates:
            groups[(item.pii_type.value, item.start, item.end)].append(item)
        result: list[Detection] = []
        for items in groups.values():
            confidence = 1.0
            for item in items:
                confidence *= 1.0 - item.confidence
            confidence = 1.0 - confidence
            best = max(items, key=lambda item: item.confidence)
            if confidence >= self.threshold:
                result.append(Detection(best.pii_type, best.start, best.end, min(confidence, 1.0), self.name, best.value, {"detectors": tuple(sorted({i.detector for i in items}))}))
        return sorted(result, key=lambda item: (item.start, item.end))
