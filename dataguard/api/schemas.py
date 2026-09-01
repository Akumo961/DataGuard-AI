from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1_000_000)
    data_location: str = Field(default="unknown", max_length=100)
    access_scope: str = Field(default="unknown", max_length=100)
    retention_days: int | None = Field(default=None, ge=0, le=36500)
    encrypted_at_rest: bool = False
    purpose_defined: bool = False
    exposure: str = Field(default="unknown", max_length=100)
    framework: str | None = Field(default="quebec_privacy", max_length=100)


class DetectionResponse(BaseModel):
    type: str
    start: int
    end: int
    confidence: float
    detector: str
    redacted_value: str


class RiskResponse(BaseModel):
    score: float
    level: str
    factors: list[dict[str, object]]
    explanation: str
    recommendations: list[str]


class GovernanceResponse(BaseModel):
    framework: str
    findings: list[dict[str, object]]


class AnalyzeResponse(BaseModel):
    analysis_id: str | None = None
    organization_id: str
    detections: list[DetectionResponse]
    risk: RiskResponse
    governance: GovernanceResponse | None = None


class PIARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_name: str = Field(min_length=1, max_length=255)
    system_description: str = Field(default="", max_length=4000)
    personal_information: list[str] = Field(default_factory=list, max_length=100)
    purposes: list[str] = Field(default_factory=list, max_length=100)
    data_sources: list[str] = Field(default_factory=list, max_length=100)
    recipients: list[str] = Field(default_factory=list, max_length=100)
    storage_locations: list[str] = Field(default_factory=list, max_length=100)
    retention: str = Field(default="", max_length=1000)
    risks: list[dict[str, object]] = Field(default_factory=list, max_length=100)
    safeguards: list[str] = Field(default_factory=list, max_length=100)


class PIAResponse(BaseModel):
    id: str
    organization_id: str
    project_name: str
    status: str
    version: int


class PIATransitionRequest(BaseModel):
    target: str = Field(min_length=1, max_length=40)
    reason: str = Field(default="", max_length=2000)


class RemediationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=4000)
    analysis_id: str | None = None
    priority: str = Field(default="MEDIUM", max_length=32)
    owner_id: str | None = Field(default=None, max_length=255)


class RemediationResponse(BaseModel):
    id: str
    organization_id: str
    status: str
    priority: str


class RemediationTransitionRequest(BaseModel):
    status: str = Field(min_length=1, max_length=32)
    evidence: dict[str, object] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str
