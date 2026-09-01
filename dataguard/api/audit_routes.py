from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from dataguard.api.dependencies import Principal, require_permission
from dataguard.audit.integrity import verify_chain_detailed
from dataguard.audit.models import AuditRecord
from dataguard.database.models import AuditEvent
from dataguard.database.session import get_session

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


@router.get("/integrity")
async def audit_integrity(
    principal: Principal = Depends(require_permission("audit:read")),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Verify the persisted audit hash chain for the caller's tenant."""
    organization_id = UUID(principal.organization_id)
    await session.execute(
        text("SELECT set_config('dataguard.organization_id', :org, true)"),
        {"org": principal.organization_id},
    )
    rows = (
        (
            await session.execute(
                select(AuditEvent)
                .where(AuditEvent.organization_id == organization_id)
                .order_by(AuditEvent.occurred_at.asc(), AuditEvent.id.asc())
            )
        )
        .scalars()
        .all()
    )
    records = [
        AuditRecord(
            event_id=str(row.id),
            timestamp=row.occurred_at.isoformat(),
            user_id=str(row.actor_id) if row.actor_id else "",
            organization_id=str(row.organization_id),
            action=row.action,
            object_type=row.object_type,
            object_id=row.object_id or "",
            previous_state=row.previous_state,
            new_state=row.new_state,
            ip_address=row.ip_address,
            request_id=row.request_id or "",
            result=row.result,
            previous_hash=row.previous_hash,
            integrity_hash=row.integrity_hash,
        )
        for row in rows
    ]
    return verify_chain_detailed(records)
