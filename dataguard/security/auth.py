"""OIDC-ready JWT authentication primitives.

Tokens are validated against configured issuer/audience and asymmetric keys in production.
Local development may use an explicitly configured HMAC secret; no insecure fallback exists.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from jwt import InvalidTokenError

from dataguard.core.config import get_settings
from dataguard.security.policy import Role, TenantContext


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    subject_id: str
    organization_id: str
    roles: frozenset[Role]

    def tenant_context(self) -> TenantContext:
        return TenantContext(self.organization_id, self.subject_id, self.roles)


def create_access_token(*, subject_id: str, organization_id: str, roles: set[Role],
                        expires_minutes: int = 15) -> str:
    settings = get_settings()
    if settings.jwt_algorithm != "HS256" or not settings.jwt_secret:
        raise RuntimeError("Local token issuance requires an explicitly configured JWT secret")
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject_id,
        "org": organization_id,
        "roles": [role.value for role in roles],
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> AuthenticatedPrincipal:
    settings = get_settings()
    if not token or len(token) > 16_384:
        raise InvalidTokenError("Invalid access token")
    if not settings.jwt_secret:
        raise RuntimeError("JWT validation key is not configured")
    options = {"require": ["sub", "org", "roles", "iat", "exp"]}
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        audience=settings.jwt_audience or None,
        issuer=settings.jwt_issuer or None,
        options=options,
    )
    subject = payload.get("sub")
    organization = payload.get("org")
    raw_roles = payload.get("roles")
    if not isinstance(subject, str) or not isinstance(organization, str) or not isinstance(raw_roles, list):
        raise InvalidTokenError("Invalid token claims")
    try:
        roles = frozenset(Role(value) for value in raw_roles)
    except ValueError as exc:
        raise InvalidTokenError("Invalid role claim") from exc
    return AuthenticatedPrincipal(subject, organization, roles)
