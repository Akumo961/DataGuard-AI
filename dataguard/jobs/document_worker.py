from __future__ import annotations

import asyncio
import os

from dataguard.jobs.document_handler import handle_document_analysis
from dataguard.jobs.worker import serve


async def main() -> None:
    redis_url = os.environ.get("DATAGUARD_REDIS_URL")
    consumer = os.environ.get("DATAGUARD_WORKER_CONSUMER", "document-worker-1")
    if not redis_url:
        raise RuntimeError("DATAGUARD_REDIS_URL is required")
    await serve(redis_url, handle_document_analysis, consumer)


if __name__ == "__main__":
    asyncio.run(main())
