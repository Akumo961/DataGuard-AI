"""JWT authentication primitives with local HMAC and OIDC/JWKS validation."""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
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


@dataclass(frozen=True)
class OIDCIdentity:
    subject: str
    email: str
    display_name: str
    organization_id: str
    roles: frozenset[Role]


@lru_cache(maxsize=8)
def _jwks_client(url: str) -> PyJWKClient:
    return PyJWKClient(
        url,
        cache_jwk_set=True,
        cache_keys=True,
        max_cached_keys=32,
        lifespan=300,
        timeout=5,
    )


def _local_signing_key() -> str | bytes | None:
    return os.getenv("DATAGUARD_JWT_SIGNING_KEY") or os.getenv("DATAGUARD_JWT_SECRET")


def _local_verification_key() -> str | bytes | None:
    return os.getenv("DATAGUARD_JWT_VERIFICATION_KEY") or _local_signing_key()


def create_access_token(
    *, subject_id: str, organization_id: str, roles: set[Role], expires_minutes: int = 15
) -> str:
    settings = get_settings()
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
    if settings.environment.lower() in {"production", "prod"}:
        key = _local_signing_key()
        if not key or settings.jwt_algorithm not in {"RS256", "ES256"}:
            raise RuntimeError("Production JWT signing key is not configured")
        if not settings.jwt_issuer or not settings.jwt_audience:
            raise RuntimeError("Production JWT issuer and audience are not configured")
        payload.update({"iss": settings.jwt_issuer, "aud": settings.jwt_audience})
        return jwt.encode(payload, key, algorithm=settings.jwt_algorithm)

    secret = (
        settings.jwt_secret.get_secret_value()
        if settings.jwt_secret
        else os.getenv("DATAGUARD_JWT_SECRET")
    )
    if settings.jwt_algorithm != "HS256" or not secret:
        raise RuntimeError("Development token issuance requires an HS256 JWT secret")
    return jwt.encode(payload, secret, algorithm="HS256")


def _oidc_key_and_claims(token: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.oidc_jwks_url or not settings.oidc_issuer_url:
        raise RuntimeError("OIDC issuer and JWKS endpoint are not configured")
    key = _jwks_client(settings.oidc_jwks_url).get_signing_key_from_jwt(token).key
    kwargs: dict[str, Any] = {
        "algorithms": ["RS256", "ES256"],
        "issuer": settings.oidc_issuer_url,
        "options": {"require": ["sub", "iss", "iat", "exp"]},
    }
    if settings.jwt_audience:
        kwargs["audience"] = settings.jwt_audience
    return jwt.decode(token, key, **kwargs)


def decode_oidc_identity(token: str) -> OIDCIdentity:
    if not token or len(token) > 16_384:
        raise InvalidTokenError("Invalid OIDC token")
    payload = _oidc_key_and_claims(token)
    subject = payload.get("sub")
    email = payload.get("email")
    organization = payload.get("org") or payload.get("org_id") or payload.get("organization_id")
    display_name = payload.get("name") or payload.get("preferred_username") or email
    raw_roles = payload.get("roles", [Role.ANALYST.value])
    if not all(isinstance(v, str) for v in (subject, email, organization, display_name)):
        raise InvalidTokenError("OIDC token is missing required identity claims")
    if not isinstance(raw_roles, list):
        raise InvalidTokenError("OIDC roles claim must be a list")
    try:
        roles = frozenset(Role(value) for value in raw_roles)
    except ValueError as exc:
        raise InvalidTokenError("OIDC token contains an unsupported role") from exc
    return OIDCIdentity(subject, email.strip().lower(), display_name, organization, roles)


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
        issuer = jwt.decode(token, options={"verify_signature": False}).get("iss")
        if issuer == settings.jwt_issuer:
            key = _local_verification_key()
            if not key:
                raise RuntimeError("Local JWT verification key is not configured")
        else:
            if not settings.oidc_jwks_url:
                raise RuntimeError("OIDC JWKS endpoint is not configured")
            key = _jwks_client(settings.oidc_jwks_url).get_signing_key_from_jwt(token).key

    options = {"require": ["sub", "roles", "iat", "exp"]}
    decode_kwargs: dict[str, Any] = {"algorithms": [settings.jwt_algorithm], "options": options}
    if settings.jwt_issuer:
        decode_kwargs["issuer"] = settings.jwt_issuer
    if settings.jwt_audience:
        decode_kwargs["audience"] = settings.jwt_audience
    payload = jwt.decode(token, key, **decode_kwargs)
    subject = payload.get("sub")
    organization = payload.get("org") or payload.get("org_id")
    raw_roles = payload.get("roles")
    jti = payload.get("jti", "")
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
