from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from dataguard.audit.models import AuditRecord


class AuditService:
    """Creates append-only records with hash chaining supplied by the persistence layer."""

    def create_record(
        self,
        *,
        event_id: str,
        timestamp: str,
        user_id: str,
        organization_id: str,
        action: str,
        object_type: str,
        object_id: str,
        previous_state: dict[str, Any] | None,
        new_state: dict[str, Any] | None,
        ip_address: str | None,
        request_id: str,
        result: str,
        previous_hash: str = "",
    ) -> AuditRecord:
        payload = {
            "event_id": event_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "organization_id": organization_id,
            "action": action,
            "object_type": object_type,
            "object_id": object_id,
            "previous_state": previous_state,
            "new_state": new_state,
            "ip_address": ip_address,
            "request_id": request_id,
            "result": result,
            "previous_hash": previous_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        integrity_hash = hashlib.sha256(canonical).hexdigest()
        return AuditRecord(**{**payload, "integrity_hash": integrity_hash})

    @staticmethod
    def verify(record: AuditRecord, previous_hash: str = "") -> bool:
        payload = asdict(record)
        actual = payload.pop("integrity_hash")
        payload["previous_hash"] = previous_hash
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(canonical).hexdigest() == actual
