from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditRecord:
    event_id: str
    timestamp: str
    user_id: str
    organization_id: str
    action: str
    object_type: str
    object_id: str
    previous_state: dict[str, Any] | None
    new_state: dict[str, Any] | None
    ip_address: str | None
    request_id: str
    result: str
    integrity_hash: str


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    organization_id: str
    evidence_type: str
    title: str
    object_type: str
    object_id: str
    framework: str | None
    framework_version: str | None
    content: dict[str, Any]
    created_at: str
    created_by: str
