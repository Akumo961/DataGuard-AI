"""Security primitives and policy boundaries for DataGuard Québec."""

from dataguard.security.auth import AuthenticatedPrincipal, create_access_token, decode_access_token
from dataguard.security.passwords import hash_password, verify_password
from dataguard.security.policy import AuthorizationPolicy, Role, TenantContext

__all__ = [
    "AuthenticatedPrincipal",
    "AuthorizationPolicy",
    "Role",
    "TenantContext",
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "verify_password",
]
