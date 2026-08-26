from datetime import datetime, timezone
from typing import Any, Protocol


class AuditSink(Protocol):
    def append(self, *, organization_id: str, actor_id: str | None, action: str,
               object_type: str, object_id: str, previous_state: dict[str, Any] | None,
               new_state: dict[str, Any] | None, request_id: str, result: str,
               occurred_at: datetime | None = None) -> None: ...


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
