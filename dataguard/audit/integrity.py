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
