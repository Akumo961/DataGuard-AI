from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Job:
    id: str
    kind: str
    payload: dict[str, Any]
    tenant_id: str


class JobQueue(ABC):
    """Bounded asynchronous job contract; implementation can use Redis or another broker."""

    @abstractmethod
    def enqueue(self, job: Job) -> None:
        raise NotImplementedError
