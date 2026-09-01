from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PIIType(StrEnum):
    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    GOVERNMENT_ID = "GOVERNMENT_ID"
    HEALTH_INSURANCE_ID = "HEALTH_INSURANCE_ID"
    PASSPORT = "PASSPORT"
    DRIVER_LICENSE = "DRIVER_LICENSE"
    HEALTH_INFORMATION = "HEALTH_INFORMATION"
    FINANCIAL_INFORMATION = "FINANCIAL_INFORMATION"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    CREDIT_CARD = "CREDIT_CARD"
    IP_ADDRESS = "IP_ADDRESS"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    EMPLOYEE_ID = "EMPLOYEE_ID"
    CUSTOMER_ID = "CUSTOMER_ID"
    TAX_ID = "TAX_ID"
    SOCIAL_INSURANCE_NUMBER = "SOCIAL_INSURANCE_NUMBER"
    BIOMETRIC_DATA = "BIOMETRIC_DATA"
    OTHER_SENSITIVE_INFORMATION = "OTHER_SENSITIVE_INFORMATION"


@dataclass(frozen=True)
class Detection:
    pii_type: PIIType
    start: int
    end: int
    confidence: float
    detector: str
    value: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskAssessment:
    score: float
    level: RiskLevel
    factors: tuple[dict[str, Any], ...]
    explanation: str
    recommendations: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisResult:
    analysis_id: str
    detections: tuple[Detection, ...]
    risk: RiskAssessment
    framework_findings: tuple[dict[str, Any], ...] = ()
