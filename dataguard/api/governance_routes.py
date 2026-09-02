from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from dataguard.api.dependencies import Principal, require_permission
from dataguard.api.schemas import (
    FindingResponse,
    PIAResponse,
    RemediationResponse,
    RemediationTransitionRequest,
)
from dataguard.database.models import Analysis, AuditEvent, Finding, PIARecord, RemediationItem
from dataguard.database.session import get_session
from dataguard.security.metrics import metrics

router = APIRouter()


def _uuid(value: str, detail: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=detail) from exc


async def _set_tenant(session: AsyncSession, organization_id: str) -> None:
    await session.execute(
        text("SELECT set_config('dataguard.organization_id', :org, true)"),
        {"org": organization_id},
    )


def _finding(row: Finding) -> FindingResponse:
    return FindingResponse(
        id=str(row.id),
        analysis_id=str(row.analysis_id),
        pii_type=row.pii_type,
        start_offset=row.start_offset,
        end_offset=row.end_offset,
        confidence=row.confidence,
        detector=row.detector,
        classification_label=row.classification_label,
        classification_confidence=row.classification_confidence,
        status=row.status,
        owner_id=row.owner_id,
        evidence=row.evidence,
    )


def _remediation(row: RemediationItem, organization_id: str) -> RemediationResponse:
    return RemediationResponse(
        id=str(row.id),
        organization_id=organization_id,
        status=row.status,
        priority=row.priority,
        owner_id=row.owner_id,
        due_at=row.due_at.isoformat() if row.due_at else None,
        verified_at=row.verified_at.isoformat() if row.verified_at else None,
        verified_by=row.verified_by,
    )


@router.get("/metrics", response_class=PlainTextResponse, tags=["observability"])
async def metrics_endpoint(
    principal: Principal = Depends(require_permission("security:manage")),
) -> PlainTextResponse:
    del principal
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@router.get("/api/v1/findings/{finding_id}", response_model=FindingResponse, tags=["findings"])
async def get_finding(
    finding_id: str,
    principal: Principal = Depends(require_permission("finding:read")),
    session: AsyncSession = Depends(get_session),
) -> FindingResponse:
    """Return a finding only when it belongs to the caller's tenant."""
    await _set_tenant(session, principal.organization_id)
    fid = _uuid(finding_id, "Invalid finding id")
    row = (
        await session.execute(
            select(Finding).where(
                Finding.id == fid,
                Finding.organization_id == UUID(principal.organization_id),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    return _finding(row)


@router.patch("/api/v1/findings/{finding_id}", response_model=FindingResponse, tags=["findings"])
async def update_finding(
    finding_id: str,
    status: str = Query(..., min_length=1, max_length=40),
    owner_id: str | None = Query(default=None, max_length=255),
    principal: Principal = Depends(require_permission("finding:manage")),
    session: AsyncSession = Depends(get_session),
) -> FindingResponse:
    await _set_tenant(session, principal.organization_id)
    fid = _uuid(finding_id, "Invalid finding id")
    normalized = status.upper()
    if normalized not in {"OPEN", "IN_REVIEW", "RESOLVED", "FALSE_POSITIVE", "ACCEPTED"}:
        raise HTTPException(status_code=400, detail="Invalid finding status")
    row = (
        await session.execute(
            select(Finding).where(
                Finding.id == fid,
                Finding.organization_id == UUID(principal.organization_id),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    previous = {"status": row.status, "owner_id": row.owner_id}
    row.status = normalized
    if owner_id is not None:
        row.owner_id = owner_id
    session.add(
        AuditEvent(
            organization_id=UUID(principal.organization_id),
            actor_id=principal.subject,
            action="FINDING_UPDATED",
            object_type="finding",
            object_id=str(row.id),
            previous_state=previous,
            new_state={"status": row.status, "owner_id": row.owner_id},
            request_id=None,
            ip_address=None,
            result="SUCCESS",
            occurred_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()
    return _finding(row)


@router.delete("/api/v1/findings/{finding_id}", status_code=204, tags=["findings"])
async def delete_finding(
    finding_id: str,
    principal: Principal = Depends(require_permission("finding:manage")),
    session: AsyncSession = Depends(get_session),
) -> None:
    await _set_tenant(session, principal.organization_id)
    fid = _uuid(finding_id, "Invalid finding id")
    row = (
        await session.execute(
            select(Finding).where(
                Finding.id == fid,
                Finding.organization_id == UUID(principal.organization_id),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    session.add(
        AuditEvent(
            organization_id=UUID(principal.organization_id),
            actor_id=principal.subject,
            action="FINDING_DELETED",
            object_type="finding",
            object_id=str(row.id),
            previous_state={"status": row.status, "pii_type": row.pii_type},
            new_state=None,
            request_id=None,
            ip_address=None,
            result="SUCCESS",
            occurred_at=datetime.now(timezone.utc),
        )
    )
    await session.delete(row)
    await session.commit()


@router.delete("/api/v1/analyses/{analysis_id}", status_code=204, tags=["analysis"])
async def delete_analysis(
    analysis_id: str,
    principal: Principal = Depends(require_permission("analysis:write")),
    session: AsyncSession = Depends(get_session),
) -> None:
    await _set_tenant(session, principal.organization_id)
    aid = _uuid(analysis_id, "Invalid analysis id")
    row = (
        await session.execute(
            select(Analysis).where(
                Analysis.id == aid,
                Analysis.organization_id == UUID(principal.organization_id),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    session.add(
        AuditEvent(
            organization_id=UUID(principal.organization_id),
            actor_id=principal.subject,
            action="ANALYSIS_DELETED",
            object_type="analysis",
            object_id=str(row.id),
            previous_state={"status": row.status, "source_type": row.source_type},
            new_state=None,
            request_id=None,
            ip_address=None,
            result="SUCCESS",
            occurred_at=datetime.now(timezone.utc),
        )
    )
    await session.delete(row)
    await session.commit()


@router.get("/api/v1/pias", response_model=list[PIAResponse], tags=["governance"])
async def list_pias(
    principal: Principal = Depends(require_permission("pia:manage")),
    session: AsyncSession = Depends(get_session),
    status: str | None = Query(default=None, max_length=40),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[PIAResponse]:
    await _set_tenant(session, principal.organization_id)
    query = select(PIARecord).where(PIARecord.organization_id == UUID(principal.organization_id))
    if status:
        query = query.where(PIARecord.status == status.upper())
    rows = (
        (
            await session.execute(
                query.order_by(PIARecord.created_at.desc()).offset(offset).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [
        PIAResponse(
            id=str(r.id),
            organization_id=principal.organization_id,
            project_name=r.project_name,
            status=r.status,
            version=r.version,
        )
        for r in rows
    ]


@router.get("/api/v1/pias/{pia_id}", response_model=PIAResponse, tags=["governance"])
async def get_pia(
    pia_id: str,
    principal: Principal = Depends(require_permission("pia:manage")),
    session: AsyncSession = Depends(get_session),
) -> PIAResponse:
    await _set_tenant(session, principal.organization_id)
    pid = _uuid(pia_id, "Invalid PIA id")
    row = (
        await session.execute(
            select(PIARecord).where(
                PIARecord.id == pid,
                PIARecord.organization_id == UUID(principal.organization_id),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="PIA not found")
    return PIAResponse(
        id=str(row.id),
        organization_id=principal.organization_id,
        project_name=row.project_name,
        status=row.status,
        version=row.version,
    )


@router.delete("/api/v1/pias/{pia_id}", status_code=204, tags=["governance"])
async def delete_pia(
    pia_id: str,
    principal: Principal = Depends(require_permission("pia:manage")),
    session: AsyncSession = Depends(get_session),
) -> None:
    await _set_tenant(session, principal.organization_id)
    pid = _uuid(pia_id, "Invalid PIA id")
    row = (
        await session.execute(
            select(PIARecord).where(
                PIARecord.id == pid,
                PIARecord.organization_id == UUID(principal.organization_id),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="PIA not found")
    session.add(
        AuditEvent(
            organization_id=UUID(principal.organization_id),
            actor_id=principal.subject,
            action="PIA_DELETED",
            object_type="pia",
            object_id=str(row.id),
            previous_state={"status": row.status, "version": row.version},
            new_state=None,
            request_id=None,
            ip_address=None,
            result="SUCCESS",
            occurred_at=datetime.now(timezone.utc),
        )
    )
    await session.delete(row)
    await session.commit()


@router.get("/api/v1/remediations", response_model=list[RemediationResponse], tags=["governance"])
async def list_remediations(
    principal: Principal = Depends(require_permission("analysis:read")),
    session: AsyncSession = Depends(get_session),
    status: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[RemediationResponse]:
    await _set_tenant(session, principal.organization_id)
    query = select(RemediationItem).where(
        RemediationItem.organization_id == UUID(principal.organization_id)
    )
    if status:
        query = query.where(RemediationItem.status == status.upper())
    rows = (
        (
            await session.execute(
                query.order_by(RemediationItem.created_at.desc()).offset(offset).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [_remediation(r, principal.organization_id) for r in rows]


@router.patch(
    "/api/v1/remediations/{remediation_id}",
    response_model=RemediationResponse,
    tags=["governance"],
)
async def transition_remediation(
    remediation_id: str,
    request: RemediationTransitionRequest,
    principal: Principal = Depends(require_permission("finding:manage")),
    session: AsyncSession = Depends(get_session),
) -> RemediationResponse:
    await _set_tenant(session, principal.organization_id)
    rid = _uuid(remediation_id, "Invalid remediation id")
    target = request.status.upper()
    if target not in {"OPEN", "IN_PROGRESS", "BLOCKED", "RESOLVED", "VERIFIED", "CLOSED"}:
        raise HTTPException(status_code=400, detail="Invalid remediation status")
    if target == "VERIFIED" and not set(principal.roles).intersection(
        {"privacy_officer", "security_admin", "org_admin"}
    ):
        raise HTTPException(status_code=403, detail="Verification requires an authorized reviewer")
    forbidden = {"raw_value", "value", "pii", "secret", "token", "password"}
    if forbidden.intersection(request.evidence):
        raise HTTPException(
            status_code=400, detail="Evidence must not contain raw sensitive values"
        )
    if target == "VERIFIED" and not request.verification_note:
        raise HTTPException(status_code=400, detail="Verification note is required")
    row = (
        await session.execute(
            select(RemediationItem).where(
                RemediationItem.id == rid,
                RemediationItem.organization_id == UUID(principal.organization_id),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Remediation not found")
    now = datetime.now(timezone.utc)
    if row.due_at is None:
        row.due_at = now + timedelta(hours=row.sla_hours)
    previous = {
        "status": row.status,
        "evidence": row.evidence,
        "verified_at": row.verified_at.isoformat() if row.verified_at else None,
    }
    row.status = target
    if request.evidence:
        row.evidence = {**row.evidence, **request.evidence}
    if request.verification_note:
        row.evidence = {**row.evidence, "verification_note": request.verification_note}
    if target == "VERIFIED":
        row.verified_at = now
        row.verified_by = principal.subject
    session.add(
        AuditEvent(
            organization_id=UUID(principal.organization_id),
            actor_id=principal.subject,
            action="REMEDIATION_VERIFIED" if target == "VERIFIED" else "REMEDIATION_TRANSITIONED",
            object_type="remediation",
            object_id=str(row.id),
            previous_state=previous,
            new_state={
                "status": row.status,
                "evidence": row.evidence,
                "verified_at": row.verified_at.isoformat() if row.verified_at else None,
                "verified_by": row.verified_by,
            },
            request_id=None,
            ip_address=None,
            result="SUCCESS",
            occurred_at=now,
        )
    )
    await session.commit()
    return _remediation(row, principal.organization_id)


@router.delete("/api/v1/remediations/{remediation_id}", status_code=204, tags=["governance"])
async def delete_remediation(
    remediation_id: str,
    principal: Principal = Depends(require_permission("finding:manage")),
    session: AsyncSession = Depends(get_session),
) -> None:
    await _set_tenant(session, principal.organization_id)
    rid = _uuid(remediation_id, "Invalid remediation id")
    row = (
        await session.execute(
            select(RemediationItem).where(
                RemediationItem.id == rid,
                RemediationItem.organization_id == UUID(principal.organization_id),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Remediation not found")
    session.add(
        AuditEvent(
            organization_id=UUID(principal.organization_id),
            actor_id=principal.subject,
            action="REMEDIATION_DELETED",
            object_type="remediation",
            object_id=str(row.id),
            previous_state={"status": row.status, "priority": row.priority},
            new_state=None,
            request_id=None,
            ip_address=None,
            result="SUCCESS",
            occurred_at=datetime.now(timezone.utc),
        )
    )
    await session.delete(row)
    await session.commit()
