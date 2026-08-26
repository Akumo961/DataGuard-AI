from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dataguard.database.models import Organization, User, UserRole
from dataguard.security.policy import Role
from dataguard.security.passwords import hash_password, verify_password

MAX_FAILED_LOGINS = 5
LOCK_MINUTES = 15


async def authenticate_local_user(session: AsyncSession, organization_id: UUID, email: str, password: str) -> User | None:
    result = await session.execute(select(User).where(User.organization_id == organization_id, User.email == email.lower()))
    user = result.scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if user is None or not user.active or (user.locked_until is not None and user.locked_until > now):
        return None
    if not user.password_hash or not verify_password(password, user.password_hash):
        user.failed_login_count += 1
        if user.failed_login_count >= MAX_FAILED_LOGINS:
            user.locked_until = now + timedelta(minutes=LOCK_MINUTES)
            user.failed_login_count = 0
        await session.flush()
        return None
    user.failed_login_count = 0
    user.locked_until = None
    await session.flush()
    return user


async def create_local_user(session: AsyncSession, organization_id: UUID, email: str, password: str, display_name: str, roles: set[Role]) -> User:
    user = User(organization_id=organization_id, email=email.lower(), password_hash=hash_password(password), display_name=display_name)
    session.add(user)
    await session.flush()
    for role in roles:
        session.add(UserRole(organization_id=organization_id, user_id=user.id, role=role.value))
    await session.flush()
    return user


async def create_organization(session: AsyncSession, slug: str, name: str) -> Organization:
    organization = Organization(slug=slug, name=name)
    session.add(organization)
    await session.flush()
    return organization
