"""JWT authentication primitives with local HMAC and OIDC/JWKS validation."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from jwt import InvalidTokenError, PyJWKClient

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
    if settings.environment == "production" or settings.jwt_algorithm != "HS256" or settings.jwt_secret is None:
        raise RuntimeError("Local token issuance is disabled for production")
    if not 1 <= expires_minutes <= 60:
        raise ValueError("Access token lifetime must be between 1 and 60 minutes")
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject_id,
        "org": organization_id,
        "roles": [role.value for role in roles],
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm="HS256")


def decode_access_token(token: str) -> AuthenticatedPrincipal:
    settings = get_settings()
    if not token or len(token) > 16_384:
        raise InvalidTokenError("Invalid access token")
    if settings.jwt_algorithm == "HS256":
        if settings.jwt_secret is None:
            raise RuntimeError("JWT validation key is not configured")
        key = settings.jwt_secret.get_secret_value()
    else:
        if not settings.oidc_jwks_url:
            raise RuntimeError("OIDC JWKS endpoint is not configured")
        key = PyJWKClient(settings.oidc_jwks_url).get_signing_key_from_jwt(token).key

    options = {"require": ["sub", "org", "roles", "iat", "exp"]}
    decode_kwargs: dict[str, Any] = {"algorithms": [settings.jwt_algorithm], "options": options}
    if settings.jwt_issuer:
        decode_kwargs["issuer"] = settings.jwt_issuer
    if settings.jwt_audience:
        decode_kwargs["audience"] = settings.jwt_audience
    payload = jwt.decode(token, key, **decode_kwargs)
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
