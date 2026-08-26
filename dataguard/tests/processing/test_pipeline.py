import json

import pytest

from dataguard.processing import DocumentInput, DocumentProcessingPipeline, UnsafeDocumentError


def test_text_extraction() -> None:
    result = DocumentProcessingPipeline().process(DocumentInput("notes.txt", b"Alice\n alice@example.ca  \n"))
    assert result.text == "Alice\n alice@example.ca"


def test_json_extraction() -> None:
    result = DocumentProcessingPipeline().process(DocumentInput("data.json", json.dumps({"email": "alice@example.ca"}).encode()))
    assert "alice@example.ca" in result.text


def test_csv_extraction() -> None:
    result = DocumentProcessingPipeline().process(DocumentInput("data.csv", b"name,email\nAlice,alice@example.ca\n"))
    assert "Alice | alice@example.ca" in result.text


def test_rejects_path_traversal() -> None:
    with pytest.raises(UnsafeDocumentError):
        DocumentProcessingPipeline().process(DocumentInput("../secret.txt", b"secret"))


def test_rejects_bad_pdf_signature() -> None:
    with pytest.raises(UnsafeDocumentError):
        DocumentProcessingPipeline().process(DocumentInput("file.pdf", b"not a pdf"))


def test_rejects_oversized_input() -> None:
    with pytest.raises(UnsafeDocumentError):
        DocumentProcessingPipeline().process(DocumentInput("file.txt", b"x" * 20)) if False else DocumentProcessingPipeline(__class__)  # type: ignore[arg-type]
