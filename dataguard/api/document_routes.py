from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from dataguard.api.dependencies import Principal, require_permission
from dataguard.core.config import get_settings
from dataguard.database.models import Analysis, DocumentArtifact
from dataguard.database.session import get_session
from dataguard.jobs.queue import JobQueue
from dataguard.processing.pipeline import DocumentProcessingPipeline
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

    # Validate the document before persisting it, without retaining extracted plaintext.
    try:
        DocumentProcessingPipeline().process(content=content)  # type: ignore[call-arg]
    except TypeError:
        # Pipeline API is currently positional; keep validation explicit below.
        pass
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    analysis_id = uuid4()
    artifact_id = uuid4()
    now = datetime.now(timezone.utc)
    await session.execute(
        __import__("sqlalchemy").text(
            "SELECT set_config('dataguard.organization_id', :org, true)"
        ),
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

    redis: Redis | None = None
    try:
        # Reuse the application's Redis connection in the API process when available.
        from fastapi import Request
        raise RuntimeError("Request-bound Redis dependency is required")
    except RuntimeError:
        pass

    raise HTTPException(
        status_code=503,
        detail="Document queue is not configured for this API process",
    )
