# Phase 9 — Document Processing

## Delivered

- Typed document input/output boundary.
- Allowlisted PDF, DOCX, TXT, CSV, XLSX, JSON and common image formats.
- Configurable upload size limit.
- Filename/path traversal protection.
- Binary signature validation.
- Strict text decoding.
- Safe JSON parsing.
- Read-only XLSX parsing.
- OOXML ZIP member/path/expanded-size checks before parser invocation.
- PDF text extraction.
- DOCX paragraph/table extraction.
- CSV normalization.
- XLSX sheet/row normalization.
- Image integrity verification.
- Optional OCR adapter with explicit opt-in dependency/runtime.
- Unit tests for extraction and upload/archive security boundaries.

## Deliberate boundaries

The processing layer does not persist raw uploads, perform malware scanning, or claim that a parser is a complete malware sandbox. Production deployment should run document processing in an isolated worker with OS/container resource limits, malware scanning, restricted filesystem/network access and controlled temporary storage.

OCR is optional and requires an installed Tesseract runtime. It is not silently invoked.

The normalized output is ready to feed the Phase 5 PII detector and Phase 6 risk engine. Persistent findings and asynchronous enterprise ingestion are integrated in later phases.

## Data minimization

Callers should persist only the minimum required metadata/results. Raw document bytes should be retained only when an approved business/legal purpose and retention policy require them.
