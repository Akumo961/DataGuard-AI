from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from dataguard.compliance.models import Applicability, ComplianceRule


@dataclass(frozen=True)
class ComplianceFinding:
    rule_id: str
    framework: str
    status: str
    severity: str
    reason: str
    required_evidence: tuple[str, ...]
    remediation: tuple[str, ...]


class ComplianceEngine:
    """Evaluates configured controls; it does not determine legal compliance."""

    def __init__(self, rules: list[ComplianceRule]) -> None:
        self.rules = tuple(rules)

    def evaluate(self, evidence: Mapping[str, bool]) -> list[ComplianceFinding]:
        findings: list[ComplianceFinding] = []
        for rule in self.rules:
            if rule.applicability is Applicability.REVIEW:
                status, reason = (
                    "REQUIRES_REVIEW",
                    "Applicability requires authorized human/legal review.",
                )
            else:
                missing = [item for item in rule.evidence_required if not evidence.get(item, False)]
                status = "PASS" if not missing else "REQUIRES_REMEDIATION"
                reason = (
                    "Configured evidence is present."
                    if not missing
                    else "Required evidence is missing: " + ", ".join(missing)
                )
            findings.append(
                ComplianceFinding(
                    rule.rule_id,
                    rule.version,
                    status,
                    rule.severity.value,
                    reason,
                    rule.evidence_required,
                    rule.remediation_recommendations,
                )
            )
        return findings
