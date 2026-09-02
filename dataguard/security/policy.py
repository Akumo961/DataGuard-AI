"""Centralized RBAC and tenant authorization policy.

The policy is intentionally independent of FastAPI so every application use case can
apply the same authorization rules.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import FrozenSet


class Role(StrEnum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    PRIVACY_OFFICER = "privacy_officer"
    SECURITY_ADMIN = "security_admin"
    ORG_ADMIN = "org_admin"


@dataclass(frozen=True)
class TenantContext:
    organization_id: str
    subject_id: str
    roles: FrozenSet[Role]

    def __post_init__(self) -> None:
        if not self.organization_id or not self.subject_id:
            raise ValueError("organization_id and subject_id are required")


class AuthorizationPolicy:
    """Fail-closed authorization policy with explicit tenant binding."""

    _permissions: dict[str, frozenset[Role]] = {
        "analysis:read": frozenset(Role),
        "analysis:write": frozenset({Role.ANALYST, Role.PRIVACY_OFFICER, Role.ORG_ADMIN}),
        "finding:read": frozenset(Role),
        "finding:manage": frozenset(
            {Role.ANALYST, Role.PRIVACY_OFFICER, Role.SECURITY_ADMIN, Role.ORG_ADMIN}
        ),
        "pii:review": frozenset(
            {Role.ANALYST, Role.PRIVACY_OFFICER, Role.SECURITY_ADMIN, Role.ORG_ADMIN}
        ),
        "pia:manage": frozenset({Role.PRIVACY_OFFICER, Role.ORG_ADMIN}),
        "classification:manage": frozenset({Role.PRIVACY_OFFICER, Role.ORG_ADMIN}),
        "security:manage": frozenset({Role.SECURITY_ADMIN, Role.ORG_ADMIN}),
        "organization:manage": frozenset({Role.ORG_ADMIN}),
        "audit:read": frozenset({Role.PRIVACY_OFFICER, Role.SECURITY_ADMIN, Role.ORG_ADMIN}),
    }

    def require(
        self, context: TenantContext, permission: str, resource_organization_id: str
    ) -> None:
        if context.organization_id != resource_organization_id:
            raise PermissionError("Cross-tenant access denied")
        allowed = self._permissions.get(permission)
        if allowed is None or not (context.roles & allowed):
            raise PermissionError("Insufficient privileges")
