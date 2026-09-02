from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from dataguard.api.dependencies import Principal, get_principal, require_permission
from dataguard.api.schemas import ClassificationPolicyRequest, ClassificationPolicyResponse
from dataguard.core.config import get_settings
from dataguard.database.models import Organization, User, UserRole
from dataguard.database.session import get_session
from dataguard.security.auth import create_access_token
from dataguard.security.passwords import hash_password, verify_password
from dataguard.security.policy import Role

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    organization_name: str = Field(min_length=2, max_length=255)
    organization_slug: str = Field(
        min_length=2, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]+$"
    )
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=1024)
    display_name: str = Field(min_length=1, max_length=255)


class LoginRequest(BaseModel):
    organization_slug: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


def _require_development() -> None:
    if get_settings().environment != "development":
        raise HTTPException(status_code=404, detail="Not found")


async def _set_tenant_context(session: AsyncSession, organization_id) -> None:
    """Bind the current transaction to the tenant before touching RLS-protected rows."""
    await session.execute(
        text("SELECT set_config('dataguard.organization_id', :org, true)"),
        {"org": str(organization_id)},
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, session: AsyncSession = Depends(get_session)) -> dict:
    _require_development()
    email = request.email.strip().lower()
    organization = Organization(
        slug=request.organization_slug, name=request.organization_name.strip()
    )
    session.add(organization)
    try:
        await session.flush()
        await _set_tenant_context(session, organization.id)
        user = User(
            organization_id=organization.id,
            email=email,
            password_hash=hash_password(request.password),
            display_name=request.display_name.strip(),
            active=True,
        )
        session.add(user)
        await session.flush()
        session.add(
            UserRole(organization_id=organization.id, user_id=user.id, role=Role.ANALYST.value)
        )
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Organization or email already exists") from exc
    return {"organization_id": str(organization.id), "user_id": str(user.id), "email": email}


@router.post("/login")
async def login(request: LoginRequest, session: AsyncSession = Depends(get_session)) -> dict:
    _require_development()
    email = request.email.strip().lower()
    organization = (
        await session.execute(
            select(Organization).where(
                Organization.slug == request.organization_slug,
                Organization.active.is_(True),
            )
        )
    ).scalar_one_or_none()
    if organization is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    await _set_tenant_context(session, organization.id)
    row = (
        await session.execute(
            select(User).where(
                User.organization_id == organization.id,
                User.email == email,
            )
        )
    ).scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if row is None or not row.active:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if row.locked_until and row.locked_until > now:
        raise HTTPException(status_code=429, detail="Account temporarily locked")
    if not row.password_hash or not verify_password(request.password, row.password_hash):
        row.failed_login_count += 1
        if row.failed_login_count >= 5:
            row.locked_until = now + timedelta(minutes=15)
        await session.commit()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    row.failed_login_count = 0
    row.locked_until = None
    roles = {
        Role(role.role)
        for role in (
            await session.execute(
                select(UserRole).where(
                    UserRole.organization_id == organization.id,
                    UserRole.user_id == row.id,
                )
            )
        ).scalars()
    }
    token = create_access_token(
        subject_id=str(row.id), organization_id=str(row.organization_id), roles=roles
    )
    await session.commit()
    return {"access_token": token, "token_type": "bearer", "expires_in": 900}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, principal: Principal = Depends(get_principal)) -> None:
    if not principal.jti:
        return
    redis: Redis | None = getattr(request.app.state, "redis", None)
    if redis is None:
        if get_settings().environment == "production":
            raise HTTPException(status_code=503, detail="Revocation service unavailable")
        return
    remaining = max(
        1,
        int(
            (datetime.fromisoformat(principal.expires_at) - datetime.now(timezone.utc)).total_seconds()
        ),
    )
    await redis.set(f"dataguard:revoked:{principal.jti}", "1", ex=remaining)


@router.get("/classification-policy", response_model=ClassificationPolicyResponse)
async def get_classification_policy(
    principal: Principal = Depends(require_permission("classification:manage")),
    session: AsyncSession = Depends(get_session),
) -> ClassificationPolicyResponse:
    row = (
        await session.execute(
            select(Organization).where(Organization.id == principal.organization_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    return ClassificationPolicyResponse(
        organization_id=principal.organization_id,
        **row.classification_policy,
    )


@router.put("/classification-policy", response_model=ClassificationPolicyResponse)
async def update_classification_policy(
    request: ClassificationPolicyRequest,
    principal: Principal = Depends(require_permission("classification:manage")),
    session: AsyncSession = Depends(get_session),
) -> ClassificationPolicyResponse:
    await _set_tenant_context(session, principal.organization_id)
    row = (
        await session.execute(
            select(Organization).where(Organization.id == principal.organization_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    policy = request.model_dump()
    row.classification_policy = policy
    await session.commit()
    return ClassificationPolicyResponse(organization_id=principal.organization_id, **policy)
