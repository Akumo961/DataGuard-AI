"""Add client provenance to immutable audit events."""

from alembic import op
import sqlalchemy as sa

revision = "20260902_0012"
down_revision = "20260901_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("audit_events", sa.Column("client", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("audit_events", "client")
