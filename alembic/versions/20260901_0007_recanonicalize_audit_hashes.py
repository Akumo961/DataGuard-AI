"""Recompute audit hashes with the canonical representation used by verification."""

from __future__ import annotations

import hashlib
import json

from alembic import op

revision = "20260901_0007"
down_revision = "20260901_0006"
branch_labels = None
depends_on = None


def _hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(canonical).hexdigest()


def upgrade() -> None:
    bind = op.get_bind()
    op.execute("ALTER TABLE audit_events DISABLE TRIGGER audit_events_append_only")
    rows = bind.execute(
        __import__("sqlalchemy").text(
            """
            SELECT id, occurred_at, actor_id, organization_id, action, object_type,
                   object_id, previous_state, new_state, ip_address, request_id, result
            FROM audit_events
            ORDER BY organization_id, occurred_at, id
            FOR UPDATE
            """
        )
    ).mappings().all()
    previous_by_org: dict[str, str] = {}
    for row in rows:
        organization_id = str(row["organization_id"])
        previous_hash = previous_by_org.get(organization_id, "")
        payload = {
            "event_id": str(row["id"]),
            "timestamp": row["occurred_at"].isoformat(),
            "user_id": str(row["actor_id"]) if row["actor_id"] else "",
            "organization_id": organization_id,
            "action": row["action"],
            "object_type": row["object_type"],
            "object_id": row["object_id"] or "",
            "previous_state": row["previous_state"],
            "new_state": row["new_state"],
            "ip_address": row["ip_address"],
            "request_id": row["request_id"] or "",
            "result": row["result"],
            "previous_hash": previous_hash,
        }
        integrity_hash = _hash(payload)
        bind.execute(
            __import__("sqlalchemy").text(
                "UPDATE audit_events SET previous_hash = :previous_hash, integrity_hash = :integrity_hash WHERE id = :id"
            ),
            {
                "id": row["id"],
                "previous_hash": previous_hash,
                "integrity_hash": integrity_hash,
            },
        )
        previous_by_org[organization_id] = integrity_hash
    op.execute("ALTER TABLE audit_events ENABLE TRIGGER audit_events_append_only")


def downgrade() -> None:
    raise RuntimeError("Audit hash recanonicalization is irreversible")
