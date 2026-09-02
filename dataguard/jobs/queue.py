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
    """Redis Streams abstraction with explicit retry, reclaim and dead-letter semantics."""

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

    async def _decode_entry(
        self, group: str, entry: tuple[str, dict]
    ) -> tuple[str, JobRecord] | None:
        message_id, fields = entry
        raw = fields.get("job")
        if raw is None:
            await self.redis.xack(self.stream, group, message_id)
            raise ValueError("Malformed queue message")
        data = json.loads(raw)
        return message_id, JobRecord(**data)

    async def claim(
        self,
        group: str,
        consumer: str,
        block_ms: int = 1000,
        reclaim_idle_ms: int = 60_000,
    ) -> tuple[str, JobRecord] | None:
        try:
            await self.redis.xgroup_create(self.stream, group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

        # Recover jobs left pending by a crashed worker. Without this, XREADGROUP('>')
        # permanently strands messages after process loss.
        try:
            reclaimed = await self.redis.xautoclaim(
                self.stream,
                group,
                consumer,
                min_idle_time=reclaim_idle_ms,
                start_id="0-0",
                count=1,
            )
            entries = reclaimed[1] if reclaimed else []
            if entries:
                return await self._decode_entry(group, entries[0])
        except (AttributeError, TypeError):
            # Older Redis clients may not expose XAUTOCLAIM; normal reads remain safe.
            pass

        rows = await self.redis.xreadgroup(
            group, consumer, {self.stream: ">"}, count=1, block=block_ms
        )
        if not rows:
            return None
        _, entries = rows[0]
        return await self._decode_entry(group, entries[0])

    async def ack(self, group: str, message_id: str) -> None:
        await self.redis.xack(self.stream, group, message_id)

    async def retry(
        self, group: str, message_id: str, job: JobRecord, max_attempts: int = 5
    ) -> None:
        next_attempt = job.attempt + 1
        if next_attempt >= max_attempts:
            await self.redis.xadd(
                self.dead_letter_stream,
                {"job": json.dumps(asdict(JobRecord(**{**asdict(job), "attempt": next_attempt})))},
            )
            await self.ack(group, message_id)
            return
        await self.redis.xadd(
            self.stream,
            {"job": json.dumps(asdict(JobRecord(**{**asdict(job), "attempt": next_attempt})))},
        )
        await self.ack(group, message_id)
