from __future__ import annotations

import mimetypes
from pathlib import PurePath

from dataguard.processing.archive import validate_ooxml_archive
from dataguard.processing.models import DocumentInput, DocumentType


class UnsafeDocumentError(ValueError):
    pass


class DocumentValidator:
    ALLOWED = {
        ".pdf": (DocumentType.PDF, b"%PDF-"),
        ".docx": (DocumentType.DOCX, b"PK\x03\x04"),
        ".txt": (DocumentType.TXT, None),
        ".csv": (DocumentType.CSV, None),
        ".xlsx": (DocumentType.XLSX, b"PK\x03\x04"),
        ".json": (DocumentType.JSON, None),
        ".png": (DocumentType.IMAGE, b"\x89PNG\r\n\x1a\n"),
        ".jpg": (DocumentType.IMAGE, b"\xff\xd8\xff"),
        ".jpeg": (DocumentType.IMAGE, b"\xff\xd8\xff"),
        ".tif": (DocumentType.IMAGE, b"II*\x00"),
        ".tiff": (DocumentType.IMAGE, b"II*\x00"),
    }

    def __init__(self, max_bytes: int = 25 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes

    def validate(self, document: DocumentInput) -> DocumentType:
        if not document.content or len(document.content) > self.max_bytes:
            raise UnsafeDocumentError("document is empty or exceeds the configured size limit")
        name = PurePath(document.filename).name
        if name != document.filename or name in {"", ".", ".."}:
            raise UnsafeDocumentError("unsafe filename")
        suffix = PurePath(name).suffix.lower()
        spec = self.ALLOWED.get(suffix)
        if spec is None:
            raise UnsafeDocumentError(f"unsupported document type: {suffix or 'none'}")
        kind, magic = spec
        if magic is not None and not document.content.startswith(magic):
            raise UnsafeDocumentError("file signature does not match the extension")
        if kind in {DocumentType.DOCX, DocumentType.XLSX}:
            try:
                validate_ooxml_archive(document.content)
            except ValueError as exc:
                raise UnsafeDocumentError(str(exc)) from exc
        if document.declared_mime and suffix in {
            ".pdf",
            ".docx",
            ".xlsx",
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
        }:
            guessed = mimetypes.guess_type(name)[0]
            if guessed and document.declared_mime != guessed:
                raise UnsafeDocumentError("declared MIME type does not match the filename")
        return kind
