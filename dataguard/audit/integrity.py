from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from dataguard.audit.models import AuditRecord


def canonical_hash(record: AuditRecord, previous_hash: str) -> str:
    payload: dict[str, Any] = asdict(record)
    payload.pop("integrity_hash", None)
    payload["previous_hash"] = previous_hash
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(canonical).hexdigest()


def verify_chain(records: Iterable[AuditRecord]) -> bool:
    previous = ""
    for record in records:
        if record.previous_hash != previous:
            return False
        if canonical_hash(record, previous) != record.integrity_hash:
            return False
        previous = record.integrity_hash
    return True


def verify_chain_detailed(records: Iterable[AuditRecord]) -> dict[str, Any]:
    """Return a machine-readable integrity result with the first broken link."""
    previous_hash = ""
    previous_event_id: str | None = None
    checked_records = 0
    for index, record in enumerate(records):
        checked_records += 1
        if record.previous_hash != previous_hash:
            return {
                "valid": False,
                "records_checked": checked_records,
                "first_broken_link": {
                    "index": index,
                    "event_id": record.event_id,
                    "previous_event_id": previous_event_id,
                    "reason": "previous_hash_mismatch",
                },
            }
        expected_hash = canonical_hash(record, previous_hash)
        if expected_hash != record.integrity_hash:
            return {
                "valid": False,
                "records_checked": checked_records,
                "first_broken_link": {
                    "index": index,
                    "event_id": record.event_id,
                    "previous_event_id": previous_event_id,
                    "reason": "integrity_hash_mismatch",
                },
            }
        previous_hash = record.integrity_hash
        previous_event_id = record.event_id
    return {
        "valid": True,
        "records_checked": checked_records,
        "first_broken_link": None,
    }
