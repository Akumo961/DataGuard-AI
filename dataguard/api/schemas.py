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
    framework: str | None = Field(default=None, max_length=100)


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


class AnalyzeResponse(BaseModel):
    organization_id: str
    detections: list[DetectionResponse]
    risk: RiskResponse


class ErrorResponse(BaseModel):
    detail: str
