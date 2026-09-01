"""Persist PIA and remediation workflow state."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260901_0004"
down_revision = "20260826_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("pia_records", "remediation_items"):
        op.create_table(
            table,
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column("organization_id", sa.Uuid(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
    op.add_column("pia_records", sa.Column("project_name", sa.String(255), nullable=False, server_default="Untitled"))
    op.add_column("pia_records", sa.Column("status", sa.String(40), nullable=False, server_default="DRAFT"))
    op.add_column("pia_records", sa.Column("version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("pia_records", sa.Column("owner_id", sa.String(255), nullable=True))
    op.add_column("pia_records", sa.Column("payload", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")))
    op.add_column("remediation_items", sa.Column("analysis_id", sa.Uuid(as_uuid=True), sa.ForeignKey("analyses.id", ondelete="SET NULL"), nullable=True))
    op.add_column("remediation_items", sa.Column("title", sa.String(255), nullable=False, server_default="Remediation"))
    op.add_column("remediation_items", sa.Column("description", sa.String(4000), nullable=False, server_default=""))
    op.add_column("remediation_items", sa.Column("status", sa.String(32), nullable=False, server_default="OPEN"))
    op.add_column("remediation_items", sa.Column("priority", sa.String(32), nullable=False, server_default="MEDIUM"))
    op.add_column("remediation_items", sa.Column("owner_id", sa.String(255), nullable=True))
    op.add_column("remediation_items", sa.Column("evidence", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")))
    for table in ("pia_records", "remediation_items"):
        op.create_index(f"ix_{table}_organization_id", table, ["organization_id"])
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY {table}_tenant_isolation ON {table} USING (organization_id = NULLIF(current_setting('dataguard.organization_id', true), '')::uuid) WITH CHECK (organization_id = NULLIF(current_setting('dataguard.organization_id', true), '')::uuid)")


def downgrade() -> None:
    for table in ("remediation_items", "pia_records"):
        op.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
        op.drop_table(table)
