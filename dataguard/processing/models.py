from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DocumentType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    IMAGE = "image"


@dataclass(frozen=True)
class DocumentInput:
    filename: str
    content: bytes
    declared_mime: str | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    filename: str
    document_type: DocumentType
    text: str
    page_count: int | None = None
    warnings: tuple[str, ...] = ()
