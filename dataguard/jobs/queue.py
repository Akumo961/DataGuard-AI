from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from uuid import uuid4

from redis.asyncio import Redis


@dataclass(frozen=True)
class JobRecord:
    """Serializable job envelope. Payload must contain identifiers, never raw customer content."""

    id: str
    kind: str
    tenant_id: str
    payload: dict
    attempt: int = 0
    created_at: float = 0.0


class JobQueue:
    """Small Redis Streams abstraction with explicit retry/dead-letter semantics."""

    def __init__(self, redis: Redis, stream: str = "dataguard:jobs") -> None:
        self.redis = redis
        self.stream = stream
        self.dead_letter_stream = f"{stream}:dead"

    async def enqueue(self, kind: str, tenant_id: str, payload: dict) -> JobRecord:
        if not kind or not tenant_id:
            raise ValueError("kind and tenant_id are required")
        record = JobRecord(
            id=str(uuid4()),
            kind=kind,
            tenant_id=tenant_id,
            payload=payload,
            created_at=time.time(),
        )
        await self.redis.xadd(
            self.stream, {"job": json.dumps(asdict(record), separators=(",", ":"))}
        )
        return record

    async def claim(
        self, group: str, consumer: str, block_ms: int = 1000
    ) -> tuple[str, JobRecord] | None:
        try:
            await self.redis.xgroup_create(self.stream, group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        rows = await self.redis.xreadgroup(
            group, consumer, {self.stream: ">"}, count=1, block=block_ms
        )
        if not rows:
            return None
        _, entries = rows[0]
        message_id, fields = entries[0]
        raw = fields.get("job")
        if raw is None:
            await self.redis.xack(self.stream, group, message_id)
            raise ValueError("Malformed queue message")
        data = json.loads(raw)
        return message_id, JobRecord(**data)

    async def ack(self, group: str, message_id: str) -> None:
        await self.redis.xack(self.stream, group, message_id)

    async def retry(
        self, group: str, message_id: str, job: JobRecord, max_attempts: int = 5
    ) -> None:
        if job.attempt + 1 >= max_attempts:
            await self.redis.xadd(
                self.dead_letter_stream,
                {
                    "job": json.dumps(
                        asdict(JobRecord(**{**asdict(job), "attempt": job.attempt + 1}))
                    )
                },
            )
            await self.ack(group, message_id)
            return
        await self.redis.xadd(
            self.stream,
            {
                "job": json.dumps(
                    asdict(JobRecord(**{**asdict(job), "attempt": job.attempt + 1}))
                )
            },
        )
        await self.ack(group, message_id)
