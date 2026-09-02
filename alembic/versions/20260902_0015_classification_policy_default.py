"""Give organizations the safe baseline classification policy by default."""

from alembic import op

revision = "20260902_0015"
down_revision = "20260902_0014"
branch_labels = None
depends_on = None

_DEFAULT_JSON = """{\"public_max\":[],\"internal_max\":[\"ORGANIZATION\"],\"confidential_max\":[\"PERSON\",\"EMAIL\",\"PHONE\",\"ADDRESS\",\"IP_ADDRESS\",\"EMPLOYEE_ID\",\"CUSTOMER_ID\"],\"restricted_max\":[\"DATE_OF_BIRTH\",\"GOVERNMENT_ID\",\"PASSPORT\",\"DRIVER_LICENSE\",\"TAX_ID\",\"FINANCIAL_INFORMATION\",\"BANK_ACCOUNT\",\"HEALTH_INFORMATION\",\"LOCATION\",\"OTHER_SENSITIVE_INFORMATION\"],\"highly_restricted_max\":[\"CREDIT_CARD\",\"SOCIAL_INSURANCE_NUMBER\",\"HEALTH_INSURANCE_ID\",\"BIOMETRIC_DATA\"]}"""


def upgrade() -> None:
    op.execute(
        "ALTER TABLE organizations "
        "ALTER COLUMN classification_policy SET DEFAULT '" + _DEFAULT_JSON.replace("'", "''") + "'::jsonb"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE organizations ALTER COLUMN classification_policy DROP DEFAULT")
