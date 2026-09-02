from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fastapi import Depends, Header, HTTPException, Request, status
from redis.asyncio import Redis

from dataguard.security.auth import decode_access_token
from dataguard.security.policy import AuthorizationPolicy, Role, TenantContext


@dataclass(frozen=True)
class Principal:
    subject: str
    organization_id: str
    roles: tuple[str, ...]
    jti: str
    expires_at: str

    def tenant_context(self) -> TenantContext:
        return TenantContext(
            self.organization_id, self.subject, frozenset(Role(role) for role in self.roles)
        )


async def get_principal(
    request: Request, authorization: str | None = Header(default=None)
) -> Principal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required"
        )
    token = authorization[7:].strip()
    try:
        principal = decode_access_token(token)
        redis: Redis | None = getattr(request.app.state, "redis", None)
        if redis is not None and await redis.exists(f"dataguard:revoked:{principal.jti}"):
            raise ValueError("Token revoked")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token"
        ) from exc
    return Principal(
        subject=principal.subject_id,
        organization_id=principal.organization_id,
        roles=tuple(sorted(role.value for role in principal.roles)),
        jti=principal.jti,
        expires_at=principal.expires_at.isoformat(),
    )


def require_permission(permission: str) -> Callable:
    policy = AuthorizationPolicy()

    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        try:
            policy.require(principal.tenant_context(), permission, principal.organization_id)
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        return principal

    return dependency
