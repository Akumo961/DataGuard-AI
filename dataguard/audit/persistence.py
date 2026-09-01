from __future__ import annotations

from sqlalchemy import Connection, event, select, text
from sqlalchemy.orm import Mapper

from dataguard.audit.integrity import canonical_hash
from dataguard.audit.models import AuditRecord
from dataguard.database.models import AuditEvent


@event.listens_for(AuditEvent, "before_insert")
def _canonicalize_persisted_audit_hash(
    mapper: Mapper[AuditEvent], connection: Connection, target: AuditEvent
) -> None:
    """Write audit hashes using the same canonical representation exposed by verification."""
    del mapper
    organization_id = str(target.organization_id)
    connection.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:org))"),
        {"org": organization_id},
    )
    previous_hash = connection.execute(
        select(AuditEvent.integrity_hash)
        .where(AuditEvent.organization_id == target.organization_id)
        .order_by(AuditEvent.occurred_at.desc(), AuditEvent.id.desc())
        .limit(1)
    ).scalar_one_or_none() or ""
    target.previous_hash = previous_hash
    record = AuditRecord(
        event_id=str(target.id),
        timestamp=target.occurred_at.isoformat(),
        user_id=str(target.actor_id) if target.actor_id else "",
        organization_id=organization_id,
        action=target.action,
        object_type=target.object_type,
        object_id=target.object_id or "",
        previous_state=target.previous_state,
        new_state=target.new_state,
        ip_address=target.ip_address,
        request_id=target.request_id or "",
        result=target.result,
        previous_hash=previous_hash,
        integrity_hash="",
    )
    target.integrity_hash = canonical_hash(record, previous_hash)
