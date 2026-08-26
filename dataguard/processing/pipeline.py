from __future__ import annotations

from dataguard.processing.extractors import CSVExtractor, DOCXExtractor, ImageExtractor, JSONExtractor, PDFExtractor, TextExtractor, XLSXExtractor
from dataguard.processing.models import DocumentInput, ExtractedDocument
from dataguard.processing.validation import DocumentValidator


class DocumentProcessingPipeline:
    def __init__(self, validator: DocumentValidator | None = None) -> None:
        self.validator = validator or DocumentValidator()
        self._extractors = {
            ".pdf": PDFExtractor(), ".docx": DOCXExtractor(), ".txt": TextExtractor(),
            ".csv": CSVExtractor(), ".xlsx": XLSXExtractor(), ".json": JSONExtractor(),
            ".png": ImageExtractor(), ".jpg": ImageExtractor(), ".jpeg": ImageExtractor(),
            ".tif": ImageExtractor(), ".tiff": ImageExtractor(),
        }

    def process(self, document: DocumentInput) -> ExtractedDocument:
        kind = self.validator.validate(document)
        suffix = document.filename.rsplit(".", 1)[-1].lower()
        extractor = self._extractors.get(f".{suffix}")
        if extractor is None:
            raise ValueError(f"no extractor for {kind.value}")
        return extractor.extract(document.filename, document.content)
