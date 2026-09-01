from dataguard.database.base import Base
from dataguard.database.models import (
    Analysis,
    APIKey,
    AuditEvent,
    RefreshToken,
    SecurityEvent,
    User,
    UserRole,
)


def test_all_phase4_tables_are_registered() -> None:
    expected = {
        "organizations",
        "users",
        "user_roles",
        "api_keys",
        "analyses",
        "security_events",
        "audit_events",
        "refresh_tokens",
    }
    assert expected.issubset(Base.metadata.tables)


def test_tenant_tables_require_organization_id() -> None:
    for model in (User, UserRole, APIKey, Analysis, SecurityEvent, AuditEvent, RefreshToken):
        assert "organization_id" in model.__table__.columns
        assert model.__table__.c.organization_id.nullable is False
