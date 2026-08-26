from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, BinaryIO


@dataclass(frozen=True)
class ExtractedDocument:
    document_id: str
    text: str
    content_type: str
    metadata: dict[str, Any]


class DocumentProcessor(ABC):
    """Safe document extraction contract.

    Implementations are responsible for bounded parsing and must not assume
    that a filename or client-provided MIME type is trustworthy.
    """

    supported_content_types: frozenset[str]

    @abstractmethod
    def extract(self, stream: BinaryIO, *, content_type: str, filename: str) -> ExtractedDocument:
        raise NotImplementedError
