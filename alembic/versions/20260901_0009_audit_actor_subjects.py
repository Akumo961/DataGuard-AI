"""Allow audit actor identifiers to represent external OIDC subjects."""
from alembic import op
import sqlalchemy as sa

revision = "20260901_0009"
down_revision = "20260901_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "audit_events",
        "actor_id",
        existing_type=sa.Uuid(as_uuid=True),
        type_=sa.String(255),
        postgresql_using="actor_id::text",
        existing_nullable=True,
    )
    op.alter_column(
        "security_events",
        "actor_id",
        existing_type=sa.Uuid(as_uuid=True),
        type_=sa.String(255),
        postgresql_using="actor_id::text",
        existing_nullable=True,
    )


def downgrade() -> None:
    # Only UUID-shaped subjects can be converted back safely. Deployments using
    # external OIDC subjects must not downgrade after those values are persisted.
    op.alter_column(
        "security_events",
        "actor_id",
        existing_type=sa.String(255),
        type_=sa.Uuid(as_uuid=True),
        postgresql_using="NULLIF(actor_id, '')::uuid",
        existing_nullable=True,
    )
    op.alter_column(
        "audit_events",
        "actor_id",
        existing_type=sa.String(255),
        type_=sa.Uuid(as_uuid=True),
        postgresql_using="NULLIF(actor_id, '')::uuid",
        existing_nullable=True,
    )
