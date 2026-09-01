from abc import ABC, abstractmethod

from dataguard.domain.models import Detection


class DetectionEngine(ABC):
    """Composable PII detection engine contract."""

    name: str

    @abstractmethod
    def detect(self, text: str) -> list[Detection]:
        raise NotImplementedError
