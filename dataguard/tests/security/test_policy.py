import pytest

from dataguard.security.policy import AuthorizationPolicy, Role, TenantContext


def test_same_tenant_role_can_access() -> None:
    context = TenantContext("org-1", "user-1", frozenset({Role.ANALYST}))
    AuthorizationPolicy().require(context, "analysis:write", "org-1")


def test_cross_tenant_access_is_denied_even_with_role() -> None:
    context = TenantContext("org-1", "user-1", frozenset({Role.ORG_ADMIN}))
    with pytest.raises(PermissionError, match="Cross-tenant"):
        AuthorizationPolicy().require(context, "analysis:write", "org-2")


def test_unknown_permission_fails_closed() -> None:
    context = TenantContext("org-1", "user-1", frozenset({Role.ORG_ADMIN}))
    with pytest.raises(PermissionError):
        AuthorizationPolicy().require(context, "unknown:permission", "org-1")
