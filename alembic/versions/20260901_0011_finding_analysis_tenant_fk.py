"""Ensure a finding's analysis belongs to the same tenant."""
from alembic import op
import sqlalchemy as sa

revision = "20260901_0011"
down_revision = "20260901_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint("uq_analysis_org_id", "analyses", ["id", "organization_id"])
    op.create_foreign_key(
        "fk_findings_analysis_tenant",
        "findings",
        "analyses",
        ["analysis_id", "organization_id"],
        ["id", "organization_id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_findings_analysis_tenant", "findings", type_="foreignkey")
    op.drop_constraint("uq_analysis_org_id", "analyses", type_="unique")
