from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PIAStatus(StrEnum):
    DRAFT = "DRAFT"
    IN_REVIEW = "IN_REVIEW"
    REQUIRES_REMEDIATION = "REQUIRES_REMEDIATION"
    APPROVED = "APPROVED"
    ARCHIVED = "ARCHIVED"


@dataclass(frozen=True)
class PIA:
    pia_id: str
    organization_id: str
    project_name: str
    system_description: str = ""
    personal_information: tuple[str, ...] = ()
    purposes: tuple[str, ...] = ()
    data_sources: tuple[str, ...] = ()
    recipients: tuple[str, ...] = ()
    storage_locations: tuple[str, ...] = ()
    retention: str = ""
    risks: tuple[dict[str, Any], ...] = ()
    safeguards: tuple[str, ...] = ()
    owner_id: str | None = None
    status: PIAStatus = PIAStatus.DRAFT
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PIAHistoryEntry:
    pia_id: str
    version: int
    from_status: PIAStatus | None
    to_status: PIAStatus
    actor_id: str
    timestamp: str
    reason: str = ""
