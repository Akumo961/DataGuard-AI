# Document Processing & Discovery

Phase 9 provides a typed document-processing boundary that validates an uploaded object before extraction.

## Supported formats

- PDF via `pypdf`
- DOCX via `python-docx`
- TXT
- CSV
- XLSX via `openpyxl`
- JSON
- PNG/JPEG/TIFF image validation via Pillow
- Optional OCR via `pytesseract` when the `ocr` extra is installed and a Tesseract runtime is explicitly deployed

## Security controls

- configurable upload-size limit (25 MiB by default)
- filename/path traversal rejection
- extension allowlist
- magic-signature checks for binary formats
- strict UTF-8 decoding for text formats
- JSON parsing rather than executable evaluation
- read-only XLSX loading
- image verification before processing
- no automatic persistence of raw uploads

Production deployments should additionally place scanning in an isolated worker/container, enforce resource/time limits, use malware scanning, and keep temporary files outside the application source tree. Those controls belong in deployment infrastructure and should not be implied by this pure extraction library.

## OCR

OCR is opt-in. The adapter does not silently fall back to OCR. This avoids unexpected compute and data-processing behavior. A production Tesseract installation must be hardened and isolated.

## Pipeline integration

`DocumentProcessingPipeline` produces normalized text and metadata. The output can be passed to the PII detection pipeline from Phase 5 and the risk engine from Phase 6. Persistent findings and asynchronous ingestion are subsequent integration concerns.
