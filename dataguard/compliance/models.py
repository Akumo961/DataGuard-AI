from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Applicability(StrEnum):
    ALWAYS = "always"
    CONDITIONAL = "conditional"
    REVIEW = "review"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ComplianceRule:
    rule_id: str
    title: str
    description: str
    category: str
    severity: Severity
    evidence_required: tuple[str, ...]
    assessment_questions: tuple[str, ...]
    remediation_recommendations: tuple[str, ...]
    version: str
    source_reference: str
    applicability: Applicability
