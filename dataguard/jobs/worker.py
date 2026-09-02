from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from redis.asyncio import Redis

from dataguard.jobs.queue import JobQueue, JobRecord

LOGGER = logging.getLogger(__name__)
JobHandler = Callable[[JobRecord], Awaitable[None]]


class AnalysisWorker:
    """Process durable jobs without placing customer content in the queue envelope."""

    def __init__(self, redis: Redis, handler: JobHandler, *, group: str = "analysis-workers") -> None:
        self.queue = JobQueue(redis)
        self.handler = handler
        self.group = group

    async def run(self, consumer: str) -> None:
        while True:
            claimed = await self.queue.claim(self.group, consumer)
            if claimed is None:
                continue
            message_id, job = claimed
            try:
                await self.handler(job)
            except Exception:
                LOGGER.exception("DataGuard job failed", extra={"job_id": job.id, "kind": job.kind})
                await self.queue.retry(self.group, message_id, job)
            else:
                await self.queue.ack(self.group, message_id)


async def serve(redis_url: str, handler: JobHandler, consumer: str) -> None:
    redis = Redis.from_url(redis_url, decode_responses=True)
    try:
        await AnalysisWorker(redis, handler).run(consumer)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    raise SystemExit("Provide a handler from the application worker entrypoint; raw documents must not be queued.")
