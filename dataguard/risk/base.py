from abc import ABC, abstractmethod
from dataguard.domain.models import Detection, RiskAssessment


class RiskEngine(ABC):
    """Transparent risk scoring contract; implementations must expose factors."""

    @abstractmethod
    def assess(self, detections: list[Detection], context: dict) -> RiskAssessment:
        raise NotImplementedError
