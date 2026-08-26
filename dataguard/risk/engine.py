from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataguard.domain.models import Detection, RiskAssessment, RiskLevel


@dataclass(frozen=True)
class RiskContext:
    data_location: str = "unknown"
    access_scope: str = "unknown"
    retention_days: int | None = None
    encrypted_at_rest: bool = False
    purpose_defined: bool = False
    exposure: str = "unknown"
    framework: str | None = None
    organization_policy_multiplier: float = 1.0


class RiskEngine:
    """Deterministic, explainable risk scoring; not a legal/compliance decision."""

    _sensitivity = {
        "PERSON": 25, "EMAIL": 25, "PHONE": 30, "ADDRESS": 30,
        "DATE_OF_BIRTH": 40, "GOVERNMENT_ID": 60, "PASSPORT": 65,
        "DRIVER_LICENSE": 60, "HEALTH_INFORMATION": 80,
        "FINANCIAL_INFORMATION": 65, "BANK_ACCOUNT": 75, "CREDIT_CARD": 80,
        "IP_ADDRESS": 15, "LOCATION": 35, "ORGANIZATION": 10,
        "EMPLOYEE_ID": 25, "CUSTOMER_ID": 30, "TAX_ID": 65,
        "SOCIAL_INSURANCE_NUMBER": 85, "BIOMETRIC_DATA": 90,
        "OTHER_SENSITIVE_INFORMATION": 70,
    }

    def assess(self, detections: list[Detection], context: RiskContext | None = None) -> RiskAssessment:
        ctx = context or RiskContext()
        factors: list[dict[str, Any]] = []
        if not detections:
            return RiskAssessment(0.0, RiskLevel.LOW, (), "No PII detections were supplied; this is not proof that no personal information exists.", ())

        base = min(70.0, max(self._sensitivity.get(d.pii_type.value, 50) for d in detections))
        quantity_points = min(15.0, len(detections) * 2.0)
        confidence = sum(d.confidence for d in detections) / len(detections)
        confidence_points = confidence * 10.0
        score = base + quantity_points + confidence_points
        factors.extend([
            {"name": "sensitivity", "points": base, "detail": "highest detected sensitivity class"},
            {"name": "quantity", "points": quantity_points, "detail": f"{len(detections)} detections"},
            {"name": "confidence", "points": confidence_points, "detail": f"mean detector confidence={confidence:.3f}"},
        ])
        if ctx.access_scope.lower() in {"public", "internet", "external"}:
            score += 10
            factors.append({"name": "access_scope", "points": 10, "detail": ctx.access_scope})
        if ctx.exposure.lower() in {"public", "external", "internet"}:
            score += 10
            factors.append({"name": "exposure", "points": 10, "detail": ctx.exposure})
        if ctx.retention_days is not None and ctx.retention_days > 365:
            score += min(8, (ctx.retention_days - 365) / 180)
            factors.append({"name": "retention", "points": min(8, (ctx.retention_days - 365) / 180), "detail": f"retention_days={ctx.retention_days}"})
        if not ctx.encrypted_at_rest:
            score += 8
            factors.append({"name": "encryption", "points": 8, "detail": "encryption at rest not confirmed"})
        if not ctx.purpose_defined:
            score += 5
            factors.append({"name": "purpose", "points": 5, "detail": "processing purpose not confirmed"})
        if ctx.data_location.lower() in {"unknown", "public"}:
            score += 4
            factors.append({"name": "data_location", "points": 4, "detail": ctx.data_location})

        score = max(0.0, min(100.0, score * max(0.0, ctx.organization_policy_multiplier)))
        level = RiskLevel.CRITICAL if score >= 80 else RiskLevel.HIGH if score >= 60 else RiskLevel.MEDIUM if score >= 30 else RiskLevel.LOW
        recommendations = self._recommendations(detections, ctx, level)
        explanation = f"Risk score {score:.1f}/100 ({level.value}) based on sensitivity, quantity, confidence and supplied control/exposure context. This score is advisory and is not a legal determination."
        return RiskAssessment(score, level, tuple(factors), explanation, tuple(recommendations))

    def _recommendations(self, detections: list[Detection], ctx: RiskContext, level: RiskLevel) -> list[str]:
        recommendations: list[str] = []
        if level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            recommendations.append("Require human privacy/security review before relying on this assessment.")
        if not ctx.encrypted_at_rest:
            recommendations.append("Confirm encryption at rest and key-management controls.")
        if not ctx.purpose_defined:
            recommendations.append("Document and validate the processing purpose.")
        if ctx.retention_days is None or ctx.retention_days > 365:
            recommendations.append("Review retention and apply the minimum period justified by the documented purpose.")
        if any(d.pii_type.value in {"HEALTH_INFORMATION", "BIOMETRIC_DATA", "SOCIAL_INSURANCE_NUMBER", "CREDIT_CARD"} for d in detections):
            recommendations.append("Apply enhanced safeguards and specialist review for highly sensitive identifiers/information.")
        return recommendations
