from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataguard.classification.base import Classification, Classifier
from dataguard.domain.models import Detection


@dataclass(frozen=True)
class ClassificationPolicy:
    """Configurable sensitivity policy used as the safe default enterprise baseline."""

    public_max: frozenset[str] = frozenset()
    internal_max: frozenset[str] = frozenset({"ORGANIZATION"})
    confidential_max: frozenset[str] = frozenset(
        {"PERSON", "EMAIL", "PHONE", "ADDRESS", "IP_ADDRESS", "EMPLOYEE_ID", "CUSTOMER_ID"}
    )
    restricted_max: frozenset[str] = frozenset(
        {
            "DATE_OF_BIRTH",
            "GOVERNMENT_ID",
            "PASSPORT",
            "DRIVER_LICENSE",
            "TAX_ID",
            "FINANCIAL_INFORMATION",
            "BANK_ACCOUNT",
            "HEALTH_INFORMATION",
            "LOCATION",
            "OTHER_SENSITIVE_INFORMATION",
        }
    )
    highly_restricted_max: frozenset[str] = frozenset(
        {"CREDIT_CARD", "SOCIAL_INSURANCE_NUMBER", "HEALTH_INSURANCE_ID", "BIOMETRIC_DATA"}
    )


class RuleBasedClassifier(Classifier):
    """Transparent classifier with explicit provenance and no legal-compliance claims."""

    name = "rule_based"

    _levels = (
        ("HIGHLY_RESTRICTED", "highly_restricted_max"),
        ("RESTRICTED", "restricted_max"),
        ("CONFIDENTIAL", "confidential_max"),
        ("INTERNAL", "internal_max"),
    )

    def __init__(self, policy: ClassificationPolicy | None = None) -> None:
        self.policy = policy or ClassificationPolicy()

    def classify(
        self, text: str, detections: list[Detection], context: dict[str, Any]
    ) -> Classification:
        del text
        if not detections:
            return Classification(
                label="PUBLIC",
                confidence=0.95,
                rationale="No sensitive-data detections were supplied; this is not proof that the content is public.",
                model_version="rules-v1",
                metadata={"detection_count": 0, "policy": "default"},
            )

        detected_types = {item.pii_type.value for item in detections}
        for label, policy_attr in self._levels:
            if detected_types & getattr(self.policy, policy_attr):
                confidence = min(
                    item.confidence for item in detections if item.pii_type.value in detected_types
                )
                return Classification(
                    label=label,
                    confidence=round(confidence, 4),
                    rationale=f"Detected sensitive categories: {', '.join(sorted(detected_types))}.",
                    model_version="rules-v1",
                    metadata={"detection_count": len(detections), "context_keys": sorted(context)},
                )
        return Classification(
            label="INTERNAL",
            confidence=0.80,
            rationale=f"Detected categories are not mapped to a higher sensitivity tier: {', '.join(sorted(detected_types))}.",
            model_version="rules-v1",
            metadata={"detection_count": len(detections), "context_keys": sorted(context)},
        )
