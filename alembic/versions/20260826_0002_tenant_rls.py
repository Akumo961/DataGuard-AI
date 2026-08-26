"""Enable fail-closed tenant isolation for tenant-scoped tables."""
from alembic import op

revision = "20260826_0002"
down_revision = "20260826_0001"
branch_labels = None
depends_on = None

TENANT_TABLES = ("users", "user_roles", "api_keys", "analyses", "security_events", "audit_events", "refresh_tokens")


def upgrade() -> None:
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(f"CREATE POLICY {table}_tenant_isolation ON {table} USING (organization_id = NULLIF(current_setting('dataguard.organization_id', true), '')::uuid) WITH CHECK (organization_id = NULLIF(current_setting('dataguard.organization_id', true), '')::uuid)")


def downgrade() -> None:
    for table in reversed(TENANT_TABLES):
        op.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
