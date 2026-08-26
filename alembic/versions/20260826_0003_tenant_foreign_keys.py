"""Bind all tenant-scoped records to an organization."""
from alembic import op

revision = "20260826_0003"
down_revision = "20260826_0002"
branch_labels = None
depends_on = None

TABLES = ("api_keys", "analyses", "security_events", "audit_events", "refresh_tokens")


def upgrade() -> None:
    for table in TABLES:
        op.create_foreign_key(f"fk_{table}_organization", table, "organizations", ["organization_id"], ["id"], ondelete="CASCADE")


def downgrade() -> None:
    for table in reversed(TABLES):
        op.drop_constraint(f"fk_{table}_organization", table, type_="foreignkey")
