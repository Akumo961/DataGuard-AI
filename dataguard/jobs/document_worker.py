from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from redis.asyncio import Redis
from sqlalchemy import delete, select, text

from dataguard.database.models import DocumentArtifact, Organization
from dataguard.database.session import SessionFactory
from dataguard.jobs.document_handler import handle_document_analysis
from dataguard.jobs.worker import AnalysisWorker


async def _retention_loop() -> None:
    while True:
        try:
            async with SessionFactory() as session:
                organizations = (await session.execute(select(Organization.id))).scalars().all()
                for organization_id in organizations:
                    await session.execute(
                        text("SELECT set_config('dataguard.organization_id', :org, true)"),
                        {"org": str(organization_id)},
                    )
                    await session.execute(
                        delete(DocumentArtifact).where(
                            DocumentArtifact.expires_at <= datetime.now(timezone.utc),
                            DocumentArtifact.organization_id == organization_id,
                        )
                    )
                await session.commit()
        except Exception:
            # A retention outage must not stop analysis processing; observability should alert operators.
            pass
        await asyncio.sleep(60)


async def main() -> None:
    redis_url = os.environ.get("DATAGUARD_REDIS_URL")
    consumer = os.environ.get("DATAGUARD_WORKER_CONSUMER", "document-worker-1")
    if not redis_url:
        raise RuntimeError("DATAGUARD_REDIS_URL is required")
    redis = Redis.from_url(redis_url, decode_responses=True)
    try:
        worker = AnalysisWorker(redis, handle_document_analysis)
        await asyncio.gather(worker.run(consumer), _retention_loop())
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
