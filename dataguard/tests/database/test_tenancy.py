from uuid import uuid4

import pytest
from sqlalchemy import select

from dataguard.database.models import Analysis
from dataguard.database.tenant import require_tenant_id, tenant_filter


def test_missing_tenant_context_fails_closed() -> None:
    with pytest.raises(PermissionError):
        require_tenant_id(None)


def test_tenant_filter_adds_organization_predicate() -> None:
    tenant = uuid4()
    statement = tenant_filter(select(Analysis), Analysis, tenant)
    compiled = str(statement.compile(compile_kwargs={"literal_binds": False}))
    assert "analyses.organization_id" in compiled


def test_non_tenant_model_is_rejected() -> None:
    with pytest.raises(TypeError):
        tenant_filter(select(Analysis), object, uuid4())
