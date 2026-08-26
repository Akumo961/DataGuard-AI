from dataguard.security.passwords import hash_password, verify_password


def test_password_hash_is_not_reversible() -> None:
    password = "A-strong-demo-password-2026!"
    hashed = hash_password(password)
    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("wrong-password", hashed)
