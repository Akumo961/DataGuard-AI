"""JWT authentication primitives with local HMAC and OIDC/JWKS validation."""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import jwt
from jwt import InvalidTokenError, PyJWKClient

from dataguard.core.config import get_settings
from dataguard.security.policy import Role, TenantContext


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    subject_id: str
    organization_id: str
    roles: frozenset[Role]
    jti: str
    expires_at: datetime

    def tenant_context(self) -> TenantContext:
        return TenantContext(self.organization_id, self.subject_id, self.roles)


def create_access_token(
    *, subject_id: str, organization_id: str, roles: set[Role], expires_minutes: int = 15
) -> str:
    settings = get_settings()
    secret = (
        settings.jwt_secret.get_secret_value()
        if settings.jwt_secret
        else os.getenv("DATAGUARD_JWT_SECRET")
    )
    if settings.environment == "production" or settings.jwt_algorithm != "HS256" or not secret:
        raise RuntimeError(
            "Local token issuance is disabled for production or missing a JWT secret"
        )
    if not 1 <= expires_minutes <= 60:
        raise ValueError("Access token lifetime must be between 1 and 60 minutes")
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject_id,
        "org": organization_id,
        "roles": [role.value for role in roles],
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
        "jti": str(uuid4()),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str) -> AuthenticatedPrincipal:
    settings = get_settings()
    if not token or len(token) > 16_384:
        raise InvalidTokenError("Invalid access token")
    if settings.jwt_algorithm == "HS256":
        key = (
            settings.jwt_secret.get_secret_value()
            if settings.jwt_secret
            else os.getenv("DATAGUARD_JWT_SECRET")
        )
        if not key:
            raise RuntimeError("JWT validation key is not configured")
    else:
        if not settings.oidc_jwks_url:
            raise RuntimeError("OIDC JWKS endpoint is not configured")
        key = PyJWKClient(settings.oidc_jwks_url).get_signing_key_from_jwt(token).key

    options = {"require": ["sub", "roles", "iat", "exp", "jti"]}
    decode_kwargs: dict[str, Any] = {"algorithms": [settings.jwt_algorithm], "options": options}
    if settings.jwt_issuer:
        decode_kwargs["issuer"] = settings.jwt_issuer
    if settings.jwt_audience:
        decode_kwargs["audience"] = settings.jwt_audience
    payload = jwt.decode(token, key, **decode_kwargs)
    subject = payload.get("sub")
    organization = payload.get("org") or payload.get("org_id")
    raw_roles = payload.get("roles")
    jti = payload.get("jti")
    expires = payload.get("exp")
    if (
        not isinstance(subject, str)
        or not isinstance(organization, str)
        or not isinstance(raw_roles, list)
        or not isinstance(jti, str)
        or not isinstance(expires, (int, float))
    ):
        raise InvalidTokenError("Invalid token claims")
    try:
        roles = frozenset(Role(value) for value in raw_roles)
    except ValueError as exc:
        raise InvalidTokenError("Invalid role claim") from exc
    return AuthenticatedPrincipal(
        subject,
        organization,
        roles,
        jti,
        datetime.fromtimestamp(expires, tz=timezone.utc),
    )
