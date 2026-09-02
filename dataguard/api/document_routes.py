from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from dataguard.api.dependencies import Principal, require_permission
from dataguard.core.config import get_settings
from dataguard.database.models import Analysis, DocumentArtifact
from dataguard.database.session import get_session
from dataguard.jobs.queue import JobQueue
from dataguard.processing.models import DocumentInput
from dataguard.processing.validation import DocumentValidator
from dataguard.security.malware import ClamAVScanner, MalwareScannerUnavailableError
from dataguard.security.document_crypto import encrypt_document

router = APIRouter(prefix="/api/v1", tags=["analysis"])


class DocumentJobResponse(BaseModel):
    analysis_id: str
    organization_id: str
    status: str


@router.post(
    "/analyze-document/async",
    response_model=DocumentJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_document_analysis(
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(require_permission("analysis:write")),
    session: AsyncSession = Depends(get_session),
) -> DocumentJobResponse:
    settings = get_settings()
    filename = (file.filename or "upload").strip()[:255]
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    if file.content_type not in settings.upload_allowed_mime_types:
        raise HTTPException(status_code=415, detail="Unsupported document type")
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Document exceeds configured size limit")

    document = DocumentInput(filename, content, file.content_type)
    try:
        DocumentValidator(
            max_bytes=settings.max_upload_bytes,
            allowed_mime_types=set(settings.upload_allowed_mime_types),
        ).validate(document)
        if settings.clamav_url:
            ClamAVScanner(settings.clamav_url).scan(content)
        elif settings.environment.lower() in {"production", "prod"}:
            raise MalwareScannerUnavailableError("Production upload scanning is not configured")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MalwareScannerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    redis: Redis | None = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Document queue unavailable")

    analysis_id = uuid4()
    artifact_id = uuid4()
    now = datetime.now(timezone.utc)
    await session.execute(
        text("SELECT set_config('dataguard.organization_id', :org, true)"),
        {"org": principal.organization_id},
    )
    analysis = Analysis(
        id=analysis_id,
        organization_id=UUID(principal.organization_id),
        source_type="document",
        source_ref=str(artifact_id),
        status="QUEUED",
        result={},
    )
    artifact = DocumentArtifact(
        id=artifact_id,
        organization_id=UUID(principal.organization_id),
        analysis_id=analysis_id,
        filename=filename,
        content_type=file.content_type,
        ciphertext=encrypt_document(
            content,
            associated_data=f"{principal.organization_id}:{artifact_id}".encode(),
        ),
        expires_at=now + timedelta(days=settings.raw_document_retention_days),
    )
    session.add_all([analysis, artifact])
    await session.commit()

    try:
        await JobQueue(redis).enqueue(
            "document_analysis",
            principal.organization_id,
            {"artifact_id": str(artifact_id), "subject_id": principal.subject},
        )
    except Exception as exc:
        analysis.status = "FAILED"
        analysis.result = {"error": "Document queue submission failed"}
        await session.commit()
        raise HTTPException(status_code=503, detail="Document queue unavailable") from exc

    return DocumentJobResponse(
        analysis_id=str(analysis_id),
        organization_id=principal.organization_id,
        status="QUEUED",
    )
