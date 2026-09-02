from dataguard.jobs.queue import JobQueue


class FakeRedis:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.acks: list[str] = []

    async def xadd(self, stream: str, fields: dict) -> str:
        self.messages.append({"stream": stream, **fields})
        return str(len(self.messages))

    async def xgroup_create(self, stream: str, group: str, id: str, mkstream: bool) -> None:
        del stream, group, id, mkstream

    async def xreadgroup(self, group: str, consumer: str, streams: dict, count: int, block: int):
        del group, consumer, streams, count, block
        return []

    async def xack(self, stream: str, group: str, message_id: str) -> None:
        self.acks.append(f"{stream}:{group}:{message_id}")


async def test_enqueue_contains_only_metadata() -> None:
    redis = FakeRedis()
    queue = JobQueue(redis)
    job = await queue.enqueue("document.analysis", "tenant-a", {"document_id": "doc-1"})
    assert job.tenant_id == "tenant-a"
    assert redis.messages[0]["stream"] == "dataguard:jobs"
    assert "document_id" in redis.messages[0]["job"]
    assert "raw_text" not in redis.messages[0]["job"]


async def test_retry_dead_letters_after_max_attempts() -> None:
    redis = FakeRedis()
    queue = JobQueue(redis)
    job = await queue.enqueue("document.analysis", "tenant-a", {"document_id": "doc-1"})
    await queue.retry("workers", "1", job, max_attempts=1)
    assert any(message["stream"] == "dataguard:jobs:dead" for message in redis.messages)
    assert redis.acks == ["dataguard:jobs:workers:1"]
