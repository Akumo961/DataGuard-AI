"""Store encrypted document inputs for durable asynchronous processing."""

from alembic import op
import sqlalchemy as sa

revision = "20260902_0016"
down_revision = "20260902_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_artifacts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("organization_id", sa.UUID(), nullable=False),
        sa.Column("analysis_id", sa.UUID(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=True),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("analysis_id"),
    )
    op.create_index(
        "ix_document_artifacts_organization_id",
        "document_artifacts",
        ["organization_id"],
    )
    op.create_index(
        "ix_document_artifacts_expires_at", "document_artifacts", ["expires_at"]
    )
    op.execute("ALTER TABLE document_artifacts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE document_artifacts FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY document_artifacts_tenant_isolation ON document_artifacts "
        "USING (organization_id::text = current_setting('dataguard.organization_id', true)) "
        "WITH CHECK (organization_id::text = current_setting('dataguard.organization_id', true))"
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS document_artifacts_tenant_isolation ON document_artifacts")
    op.drop_index("ix_document_artifacts_expires_at", table_name="document_artifacts")
    op.drop_index("ix_document_artifacts_organization_id", table_name="document_artifacts")
    op.drop_table("document_artifacts")
