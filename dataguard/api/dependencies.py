from __future__ import annotations

import os
from dataclasses import dataclass

import jwt
from fastapi import Header, HTTPException, status


@dataclass(frozen=True)
class Principal:
    subject: str
    organization_id: str
    roles: tuple[str, ...]


def get_principal(authorization: str | None = Header(default=None)) -> Principal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    token = authorization[7:].strip()
    secret = os.getenv("DATAGUARD_JWT_SECRET")
    algorithm = os.getenv("DATAGUARD_JWT_ALGORITHM", "HS256")
    if not secret:
        raise HTTPException(status_code=503, detail="authentication provider is not configured")
    try:
        claims = jwt.decode(token, secret, algorithms=[algorithm], options={"require": ["sub", "org_id", "exp"]})
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="invalid access token") from exc
    subject = str(claims["sub"])
    organization_id = str(claims["org_id"])
    raw_roles = claims.get("roles", [])
    roles = tuple(str(role) for role in raw_roles) if isinstance(raw_roles, list) else ()
    return Principal(subject, organization_id, roles)
