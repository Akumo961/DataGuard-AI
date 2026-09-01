"""Add persisted audit hash-chain fields."""

from alembic import op
import sqlalchemy as sa

revision = "20260901_0005"
down_revision = "20260901_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "audit_events",
        sa.Column("previous_hash", sa.String(length=64), nullable=False, server_default=""),
    )
    op.add_column("audit_events", sa.Column("integrity_hash", sa.String(length=64), nullable=True))

    # Backfill legacy rows deterministically in occurred_at/id order. The chain is
    # meaningful from this migration forward; legacy rows receive a valid chain.
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT id, organization_id, actor_id, action, object_type, object_id, "
            "previous_state, new_state, request_id, ip_address, result, occurred_at "
            "FROM audit_events ORDER BY organization_id, occurred_at, id"
        )
    ).mappings().all()
    import hashlib
    import json

    previous_by_org: dict[str, str] = {}
    for row in rows:
        org = str(row["organization_id"])
        previous = previous_by_org.get(org, "")
        payload = {
            "event_id": str(row["id"]),
            "timestamp": row["occurred_at"].isoformat() if row["occurred_at"] else "",
            "user_id": str(row["actor_id"]) if row["actor_id"] else "",
            "organization_id": org,
            "action": row["action"],
            "object_type": row["object_type"],
            "object_id": row["object_id"] or "",
            "previous_state": row["previous_state"],
            "new_state": row["new_state"],
            "ip_address": row["ip_address"],
            "request_id": row["request_id"] or "",
            "result": row["result"],
            "previous_hash": previous,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        integrity = hashlib.sha256(canonical).hexdigest()
        bind.execute(
            sa.text(
                "UPDATE audit_events SET previous_hash=:previous_hash, integrity_hash=:integrity_hash "
                "WHERE id=:id"
            ),
            {"previous_hash": previous, "integrity_hash": integrity, "id": row["id"]},
        )
        previous_by_org[org] = integrity

    op.alter_column("audit_events", "integrity_hash", nullable=False)
    op.alter_column("audit_events", "previous_hash", server_default=None)
    op.create_index(
        "ix_audit_events_org_occurred_at", "audit_events", ["organization_id", "occurred_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_audit_events_org_occurred_at", table_name="audit_events")
    op.drop_column("audit_events", "integrity_hash")
    op.drop_column("audit_events", "previous_hash")
