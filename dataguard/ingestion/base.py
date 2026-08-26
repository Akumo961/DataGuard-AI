from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    filename: str
    content_type: str
    size_bytes: int
    path: Path | None = None


class Connector(ABC):
    """Read-only discovery connector contract.

    Connectors return metadata and controlled document references. They must not
    bypass tenant authorization, retention policy, or SSRF/path validation.
    """

    name: str

    @abstractmethod
    def discover(self) -> list[SourceDocument]:
        raise NotImplementedError

    @abstractmethod
    def open(self, document: SourceDocument):
        raise NotImplementedError
