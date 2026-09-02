import base64

import pytest

from dataguard.security.document_crypto import decrypt_document, encrypt_document


def test_document_encryption_round_trip(monkeypatch):
    monkeypatch.setenv("DATAGUARD_ENVIRONMENT", "test")
    monkeypatch.setenv(
        "DATAGUARD_DOCUMENT_ENCRYPTION_KEY",
        base64.urlsafe_b64encode(b"x" * 32).decode("ascii"),
    )
    blob = encrypt_document(b"secret document", associated_data=b"tenant:artifact")
    assert blob != b"secret document"
    assert decrypt_document(blob, associated_data=b"tenant:artifact") == b"secret document"


def test_document_encryption_rejects_tampering(monkeypatch):
    monkeypatch.setenv("DATAGUARD_ENVIRONMENT", "test")
    monkeypatch.setenv(
        "DATAGUARD_DOCUMENT_ENCRYPTION_KEY",
        base64.urlsafe_b64encode(b"y" * 32).decode("ascii"),
    )
    blob = bytearray(encrypt_document(b"secret document", associated_data=b"tenant:artifact"))
    blob[-1] ^= 1
    with pytest.raises(Exception):
        decrypt_document(bytes(blob), associated_data=b"tenant:artifact")
