from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import AsyncIterator


class ConnectorType(StrEnum):
    OBJECT_STORAGE = "object_storage"
    DATABASE = "database"
    DOCUMENT = "document"
    SIEM = "siem"
    IAM = "iam"
    TICKETING = "ticketing"


@dataclass(frozen=True)
class ConnectorObject:
    """Metadata-only source object; content is streamed and never stored in connector state."""

    object_id: str
    name: str
    content_type: str | None
    size: int | None
    modified_at: str | None
    source_uri: str


class Connector(ABC):
    """Least-privilege connector boundary. Implementations must scope access to one tenant."""

    type: ConnectorType

    @abstractmethod
    async def health(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def list_objects(self, prefix: str | None = None) -> AsyncIterator[ConnectorObject]:
        raise NotImplementedError

    @abstractmethod
    async def read_object(self, object_id: str) -> AsyncIterator[bytes]:
        raise NotImplementedError
