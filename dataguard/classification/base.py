from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from dataguard.domain.models import Detection


@dataclass(frozen=True)
class Classification:
    label: str
    confidence: float
    rationale: str
    model_version: str | None = None
    metadata: dict[str, Any] | None = None


class Classifier(ABC):
    """Replaceable sensitivity classifier contract.

    Implementations may be deterministic rules or ML models. They must expose
    provenance and confidence and must not make legal-compliance conclusions.
    """

    name: str

    @abstractmethod
    def classify(
        self, text: str, detections: list[Detection], context: dict[str, Any]
    ) -> Classification:
        raise NotImplementedError
