"""Initial enterprise schema.

Revision ID: 20260826_0001
Revises:
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260826_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    uuid = postgresql.UUID(as_uuid=True)
    jsonb = postgresql.JSONB()
    op.create_table("organizations", sa.Column("id", uuid, primary_key=True), sa.Column("slug", sa.String(100), nullable=False, unique=True), sa.Column("name", sa.String(255), nullable=False), sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_table("users", sa.Column("id", uuid, primary_key=True), sa.Column("organization_id", uuid, sa.ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False), sa.Column("external_subject", sa.String(255)), sa.Column("email", sa.String(320), nullable=False), sa.Column("password_hash", sa.String(512)), sa.Column("display_name", sa.String(255), nullable=False), sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()), sa.Column("failed_login_count", sa.Integer(), nullable=False, server_default="0"), sa.Column("locked_until", sa.DateTime(timezone=True)), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.UniqueConstraint("organization_id", "email", name="uq_users_org_email"))
    op.create_index("ix_users_organization_id", "users", ["organization_id"])
    op.create_table("user_roles", sa.Column("id", uuid, primary_key=True), sa.Column("organization_id", uuid, sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False), sa.Column("user_id", uuid, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("role", sa.String(64), nullable=False), sa.UniqueConstraint("organization_id", "user_id", "role", name="uq_user_roles"))
    op.create_index("ix_user_roles_organization_id", "user_roles", ["organization_id"])
    op.create_table("api_keys", sa.Column("id", uuid, primary_key=True), sa.Column("organization_id", uuid, nullable=False), sa.Column("name", sa.String(255), nullable=False), sa.Column("key_prefix", sa.String(32), nullable=False), sa.Column("key_hash", sa.String(128), nullable=False, unique=True), sa.Column("revoked_at", sa.DateTime(timezone=True)), sa.Column("expires_at", sa.DateTime(timezone=True)), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_api_keys_organization_id", "api_keys", ["organization_id"])
    op.create_table("analyses", sa.Column("id", uuid, primary_key=True), sa.Column("organization_id", uuid, nullable=False), sa.Column("status", sa.String(32), nullable=False), sa.Column("source_type", sa.String(64), nullable=False), sa.Column("source_ref", sa.String(512)), sa.Column("result", jsonb, nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_analyses_organization_id", "analyses", ["organization_id"])
    op.create_table("security_events", sa.Column("id", uuid, primary_key=True), sa.Column("organization_id", uuid, nullable=False), sa.Column("event_type", sa.String(100), nullable=False), sa.Column("actor_id", uuid), sa.Column("request_id", sa.String(128)), sa.Column("ip_address", sa.String(64)), sa.Column("details", jsonb, nullable=False), sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_security_events_organization_id", "security_events", ["organization_id"])
    op.create_table("audit_events", sa.Column("id", uuid, primary_key=True), sa.Column("organization_id", uuid, nullable=False), sa.Column("actor_id", uuid), sa.Column("action", sa.String(100), nullable=False), sa.Column("object_type", sa.String(100), nullable=False), sa.Column("object_id", sa.String(255)), sa.Column("previous_state", jsonb), sa.Column("new_state", jsonb), sa.Column("request_id", sa.String(128)), sa.Column("ip_address", sa.String(64)), sa.Column("result", sa.String(32), nullable=False), sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_audit_events_organization_id", "audit_events", ["organization_id"])
    op.create_table("refresh_tokens", sa.Column("id", uuid, primary_key=True), sa.Column("organization_id", uuid, nullable=False), sa.Column("user_id", uuid, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("token_hash", sa.String(128), nullable=False, unique=True), sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False), sa.Column("revoked_at", sa.DateTime(timezone=True)), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_refresh_tokens_organization_id", "refresh_tokens", ["organization_id"])


def downgrade() -> None:
    for table in ("refresh_tokens", "audit_events", "security_events", "analyses", "api_keys", "user_roles", "users", "organizations"):
        op.drop_table(table)
