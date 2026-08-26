"""Password hashing using Argon2id through pwdlib."""

from pwdlib import PasswordHash

_password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    if not password or len(password) > 1024:
        raise ValueError("Password must contain 1-1024 characters")
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if not password or not password_hash:
        return False
    try:
        return _password_hash.verify(password, password_hash)
    except (TypeError, ValueError):
        return False
