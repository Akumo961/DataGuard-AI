from __future__ import annotations

from dataguard.core.config import get_settings
from dataguard.processing.extractors import (
    CSVExtractor,
    DOCXExtractor,
    ImageExtractor,
    JSONExtractor,
    PDFExtractor,
    TextExtractor,
    XLSXExtractor,
)
from dataguard.processing.models import DocumentInput, ExtractedDocument
from dataguard.processing.validation import DocumentValidator
from dataguard.security.malware import ClamAVScanner, MalwareScannerUnavailableError


class DocumentProcessingPipeline:
    def __init__(self, validator: DocumentValidator | None = None) -> None:
        settings = get_settings()
        self.validator = validator or DocumentValidator(
            max_bytes=settings.max_upload_bytes,
            allowed_mime_types=set(settings.upload_allowed_mime_types),
        )
        self.scanner = ClamAVScanner(settings.clamav_url) if settings.clamav_url else None
        self._extractors = {
            ".pdf": PDFExtractor(),
            ".docx": DOCXExtractor(),
            ".txt": TextExtractor(),
            ".csv": CSVExtractor(),
            ".xlsx": XLSXExtractor(),
            ".json": JSONExtractor(),
            ".png": ImageExtractor(),
            ".jpg": ImageExtractor(),
            ".jpeg": ImageExtractor(),
            ".tif": ImageExtractor(),
            ".tiff": ImageExtractor(),
        }

    def process(self, document: DocumentInput) -> ExtractedDocument:
        kind = self.validator.validate(document)
        if self.scanner is not None:
            self.scanner.scan(document.content)
        elif get_settings().environment.lower() in {"production", "prod"}:
            raise MalwareScannerUnavailableError("Production upload scanning is not configured")
        suffix = document.filename.rsplit(".", 1)[-1].lower()
        extractor = self._extractors.get(f".{suffix}")
        if extractor is None:
            raise ValueError(f"no extractor for {kind.value}")
        return extractor.extract(document.filename, document.content)
