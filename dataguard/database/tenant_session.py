from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@asynccontextmanager
async def tenant_transaction(
    session: AsyncSession, organization_id: UUID
) -> AsyncIterator[AsyncSession]:
    """Run work with a transaction-local tenant setting used by PostgreSQL RLS."""
    if organization_id is None:
        raise PermissionError("Tenant context is required")
    async with session.begin():
        await session.execute(
            text("select set_config('dataguard.organization_id', :tenant, true)"),
            {"tenant": str(organization_id)},
        )
        yield session
