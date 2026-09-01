from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from dataguard.security.auth import decode_access_token


@dataclass(frozen=True)
class Principal:
    subject: str
    organization_id: str
    roles: tuple[str, ...]


def get_principal(authorization: str | None = Header(default=None)) -> Principal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    token = authorization[7:].strip()
    try:
        principal = decode_access_token(token)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token") from exc
    return Principal(
        subject=principal.subject_id,
        organization_id=principal.organization_id,
        roles=tuple(sorted(role.value for role in principal.roles)),
    )
