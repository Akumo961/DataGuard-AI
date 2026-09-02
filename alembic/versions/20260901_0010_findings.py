"""Persist analysis findings without storing raw detected values."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260901_0010"
down_revision = "20260901_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "findings",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "organization_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "analysis_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("analyses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pii_type", sa.String(80), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("detector", sa.String(80), nullable=False),
        sa.Column("classification_label", sa.String(40), nullable=False),
        sa.Column("classification_confidence", sa.Float(), nullable=False),
        sa.Column("status", sa.String(40), nullable=False, server_default="OPEN"),
        sa.Column("owner_id", sa.String(255), nullable=True),
        sa.Column("evidence", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_findings_organization_id", "findings", ["organization_id"])
    op.create_index("ix_findings_analysis_id", "findings", ["analysis_id"])
    op.create_index("ix_findings_pii_type", "findings", ["pii_type"])
    op.execute("ALTER TABLE findings ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE findings FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY findings_tenant_isolation ON findings "
        "USING (organization_id = NULLIF(current_setting('dataguard.organization_id', true), '')::uuid) "
        "WITH CHECK (organization_id = NULLIF(current_setting('dataguard.organization_id', true), '')::uuid)"
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS findings_tenant_isolation ON findings")
    op.execute("ALTER TABLE findings NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE findings DISABLE ROW LEVEL SECURITY")
    op.drop_index("ix_findings_pii_type", table_name="findings")
    op.drop_index("ix_findings_analysis_id", table_name="findings")
    op.drop_index("ix_findings_organization_id", table_name="findings")
    op.drop_table("findings")
