from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from dataguard.pia.models import PIA, PIAHistoryEntry, PIAStatus

_ALLOWED: dict[PIAStatus, frozenset[PIAStatus]] = {
    PIAStatus.DRAFT: frozenset({PIAStatus.IN_REVIEW}),
    PIAStatus.IN_REVIEW: frozenset(
        {PIAStatus.REQUIRES_REMEDIATION, PIAStatus.APPROVED, PIAStatus.DRAFT}
    ),
    PIAStatus.REQUIRES_REMEDIATION: frozenset({PIAStatus.IN_REVIEW}),
    PIAStatus.APPROVED: frozenset({PIAStatus.ARCHIVED}),
    PIAStatus.ARCHIVED: frozenset(),
}


class PIAWorkflow:
    def transition(
        self, pia: PIA, target: PIAStatus, actor_id: str, reason: str = ""
    ) -> tuple[PIA, PIAHistoryEntry]:
        if target not in _ALLOWED[pia.status]:
            raise ValueError(f"invalid PIA transition: {pia.status} -> {target}")
        if not actor_id:
            raise ValueError("actor_id is required")
        updated = replace(pia, status=target, version=pia.version + 1)
        history = PIAHistoryEntry(
            pia_id=pia.pia_id,
            version=updated.version,
            from_status=pia.status,
            to_status=target,
            actor_id=actor_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )
        return updated, history
