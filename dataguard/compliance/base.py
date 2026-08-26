from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlFinding:
    framework: str
    rule_id: str
    status: str
    severity: str
    evidence_required: tuple[str, ...]
    remediation: tuple[str, ...]
    legal_verification_required: bool = True


class ComplianceEngine(ABC):
    """Versioned control evaluation contract; never emits legal conclusions."""

    @abstractmethod
    def evaluate(self, context: dict) -> list[ControlFinding]:
        raise NotImplementedError
