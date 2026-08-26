"""Non-reversible API-key primitives.

Only a digest is persisted by a future database adapter. The plaintext key is returned once
at provisioning time and must never be logged.
"""

import hashlib
import hmac
import secrets

_PREFIX = "dgk_"


def generate_api_key() -> tuple[str, str]:
    secret = secrets.token_urlsafe(32)
    plaintext = f"{_PREFIX}{secret}"
    return plaintext, hash_api_key(plaintext)


def hash_api_key(api_key: str) -> str:
    if not api_key.startswith(_PREFIX) or len(api_key) > 128:
        raise ValueError("Invalid API key")
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_api_key(api_key: str, stored_digest: str) -> bool:
    try:
        candidate = hash_api_key(api_key)
    except ValueError:
        return False
    return hmac.compare_digest(candidate, stored_digest)
