"""Add organization-scoped classification policy storage."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260902_0014"
down_revision = "20260902_0013"
branch_labels = None
depends_on = None


_DEFAULT = {
    "public_max": [],
    "internal_max": ["ORGANIZATION"],
    "confidential_max": [
        "PERSON",
        "EMAIL",
        "PHONE",
        "ADDRESS",
        "IP_ADDRESS",
        "EMPLOYEE_ID",
        "CUSTOMER_ID",
    ],
    "restricted_max": [
        "DATE_OF_BIRTH",
        "GOVERNMENT_ID",
        "PASSPORT",
        "DRIVER_LICENSE",
        "TAX_ID",
        "FINANCIAL_INFORMATION",
        "BANK_ACCOUNT",
        "HEALTH_INFORMATION",
        "LOCATION",
        "OTHER_SENSITIVE_INFORMATION",
    ],
    "highly_restricted_max": [
        "CREDIT_CARD",
        "SOCIAL_INSURANCE_NUMBER",
        "HEALTH_INSURANCE_ID",
        "BIOMETRIC_DATA",
    ],
}


def upgrade() -> None:
    op.add_column(
        "organizations",
        sa.Column("classification_policy", postgresql.JSONB(), nullable=True),
    )
    op.execute(
        sa.text("UPDATE organizations SET classification_policy = :policy WHERE classification_policy IS NULL").bindparams(policy=_DEFAULT)
    )
    op.alter_column("organizations", "classification_policy", nullable=False)


def downgrade() -> None:
    op.drop_column("organizations", "classification_policy")
