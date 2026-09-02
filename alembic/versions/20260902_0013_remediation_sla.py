"""Persist remediation SLA and verification state."""

from alembic import op
import sqlalchemy as sa

revision = "20260902_0013"
down_revision = "20260902_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("remediation_items", sa.Column("sla_hours", sa.Integer(), nullable=True))
    op.add_column("remediation_items", sa.Column("due_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("remediation_items", sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("remediation_items", sa.Column("verified_by", sa.String(length=255), nullable=True))
    op.execute("UPDATE remediation_items SET sla_hours = 168 WHERE sla_hours IS NULL")
    op.alter_column("remediation_items", "sla_hours", nullable=False, server_default="168")


def downgrade() -> None:
    op.drop_column("remediation_items", "verified_by")
    op.drop_column("remediation_items", "verified_at")
    op.drop_column("remediation_items", "due_at")
    op.drop_column("remediation_items", "sla_hours")
