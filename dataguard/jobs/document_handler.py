from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, text

from dataguard.api.app import _analysis_payload, _governance
from dataguard.api.schemas import AnalyzeRequest
from dataguard.classification.rules import RuleBasedClassifier
from dataguard.database.models import Analysis, AuditEvent, DocumentArtifact, Finding, Organization
from dataguard.database.session import SessionFactory
from dataguard.detection.ensemble import EnsembleDetector
from dataguard.detection.pipeline import PIIDetectionPipeline
from dataguard.detection.regex import RegexPIIDetector
from dataguard.jobs.queue import JobRecord
from dataguard.processing.models import DocumentInput
from dataguard.processing.pipeline import DocumentProcessingPipeline
from dataguard.risk.engine import RiskContext, RiskEngine
from dataguard.security.audit_context import set_classification_policy
from dataguard.security.document_crypto import decrypt_document


async def handle_document_analysis(job: JobRecord) -> None:
    if job.kind != "document_analysis":
        raise ValueError(f"Unsupported job kind: {job.kind}")
    artifact_id = job.payload.get("artifact_id")
    subject_id = job.payload.get("subject_id")
    if not isinstance(artifact_id, str) or not isinstance(subject_id, str):
        raise ValueError("document_analysis requires artifact_id and subject_id")

    async with SessionFactory() as session:
        await session.execute(
            text("SELECT set_config('dataguard.organization_id', :org, true)"),
            {"org": job.tenant_id},
        )
        organization = (
            await session.execute(
                select(Organization).where(Organization.id == UUID(job.tenant_id))
            )
        ).scalar_one_or_none()
        if organization is None or not organization.active:
            raise ValueError("Document tenant is not active")
        set_classification_policy(organization.classification_policy)
        artifact = (
            await session.execute(
                select(DocumentArtifact).where(
                    DocumentArtifact.id == UUID(artifact_id),
                    DocumentArtifact.organization_id == UUID(job.tenant_id),
                )
            )
        ).scalar_one_or_none()
        if artifact is None:
            raise ValueError("Document artifact not found")
        analysis = (
            await session.execute(
                select(Analysis).where(
                    Analysis.id == artifact.analysis_id,
                    Analysis.organization_id == UUID(job.tenant_id),
                )
            )
        ).scalar_one_or_none()
        if analysis is None:
            raise ValueError("Analysis not found")
        if analysis.status == "COMPLETED":
            return
        if artifact.expires_at <= datetime.now(timezone.utc):
            analysis.status = "FAILED"
            analysis.result = {"error": "Document retention window expired"}
            await session.commit()
            raise ValueError("Document artifact expired")

        analysis.status = "PROCESSING"
        await session.commit()
        try:
            content = decrypt_document(
                artifact.ciphertext,
                associated_data=f"{job.tenant_id}:{artifact.id}".encode(),
            )
            extracted = DocumentProcessingPipeline().process(
                DocumentInput(artifact.filename, content, artifact.content_type)
            )
            pipeline = PIIDetectionPipeline(EnsembleDetector([RegexPIIDetector()]))
            detections = pipeline.detect(extracted.text)
            classification = RuleBasedClassifier().classify(
                extracted.text,
                detections,
                {"organization_id": job.tenant_id, "document_type": extracted.document_type.value},
            )
            risk = RiskEngine().assess(detections, RiskContext())
            governance = _governance(extracted.text, AnalyzeRequest(text=extracted.text))
            payload = _analysis_payload(detections, classification, risk, governance)
            payload["document"] = {
                "filename": extracted.filename,
                "document_type": extracted.document_type.value,
                "page_count": extracted.page_count,
                "warnings": list(extracted.warnings),
            }
            analysis.status = "COMPLETED"
            analysis.result = payload
            session.add_all(
                Finding(
                    organization_id=UUID(job.tenant_id),
                    analysis_id=analysis.id,
                    pii_type=item["type"],
                    start_offset=item["start"],
                    end_offset=item["end"],
                    confidence=item["confidence"],
                    detector=item["detector"],
                    classification_label=payload["classification"]["label"],
                    classification_confidence=payload["classification"]["confidence"],
                    status="OPEN",
                    owner_id=subject_id,
                    evidence={
                        "redacted_value": item["redacted_value"],
                        "detector": item["detector"],
                    },
                )
                for item in payload["detections"]
            )
            artifact.processed_at = datetime.now(timezone.utc)
            session.add(
                AuditEvent(
                    organization_id=UUID(job.tenant_id),
                    actor_id=subject_id,
                    action="ANALYSIS_COMPLETED",
                    object_type="analysis",
                    object_id=str(analysis.id),
                    previous_state={"status": "PROCESSING"},
                    new_state={"status": "COMPLETED", "source_type": "document"},
                    request_id=None,
                    ip_address=None,
                    result="SUCCESS",
                    occurred_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        except Exception:
            await session.rollback()
            analysis.status = "FAILED"
            analysis.result = {"error": "Document analysis failed"}
            await session.commit()
            raise
