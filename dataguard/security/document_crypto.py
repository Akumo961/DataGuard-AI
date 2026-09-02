from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_KEY_ENV = "DATAGUARD_DOCUMENT_ENCRYPTION_KEY"
_DEV_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


def _key() -> bytes:
    raw = os.getenv(_KEY_ENV)
    if not raw:
        raise RuntimeError(f"{_KEY_ENV} is required for durable document processing")
    if os.getenv("DATAGUARD_ENVIRONMENT", "development").lower() in {"production", "prod"}:
        if raw == _DEV_KEY:
            raise RuntimeError(
                "The development document encryption key cannot be used in production"
            )
    try:
        key = base64.urlsafe_b64decode(raw.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise RuntimeError(f"{_KEY_ENV} must be URL-safe base64") from exc
    if len(key) not in {16, 24, 32}:
        raise RuntimeError(f"{_KEY_ENV} must decode to 16, 24, or 32 bytes")
    return key


def encrypt_document(content: bytes, *, associated_data: bytes) -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(_key()).encrypt(nonce, content, associated_data)


def decrypt_document(blob: bytes, *, associated_data: bytes) -> bytes:
    if len(blob) < 13:
        raise ValueError("Encrypted document is malformed")
    return AESGCM(_key()).decrypt(blob[:12], blob[12:], associated_data)
