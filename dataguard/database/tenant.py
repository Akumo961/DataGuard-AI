from __future__ import annotations

from uuid import UUID

from sqlalchemy import Select


def require_tenant_id(organization_id: UUID | None) -> UUID:
    if organization_id is None:
        raise PermissionError("Tenant context is required")
    return organization_id


def tenant_filter(statement: Select, model: type, organization_id: UUID | None) -> Select:
    tenant_id = require_tenant_id(organization_id)
    if not hasattr(model, "organization_id"):
        raise TypeError(f"{model.__name__} is not tenant scoped")
    return statement.where(model.organization_id == tenant_id)
