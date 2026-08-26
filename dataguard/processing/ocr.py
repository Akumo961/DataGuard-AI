from __future__ import annotations

from dataguard.processing.models import DocumentType, ExtractedDocument


class OCRUnavailableError(RuntimeError):
    pass


class OCRExtractor:
    """Optional OCR adapter; requires pytesseract and an installed Tesseract binary."""

    def extract(self, filename: str, content: bytes) -> ExtractedDocument:
        try:
            import io
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise OCRUnavailableError("install the 'ocr' extra to enable OCR") from exc
        try:
            image = Image.open(io.BytesIO(content))
            image.verify()
            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image)
        except Exception as exc:
            raise OCRUnavailableError("OCR processing failed") from exc
        return ExtractedDocument(filename, DocumentType.IMAGE, text.strip())
