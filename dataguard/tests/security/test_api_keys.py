from dataguard.security.api_keys import generate_api_key, verify_api_key


def test_api_key_is_verifiable_without_persisting_plaintext() -> None:
    plaintext, digest = generate_api_key()
    assert plaintext.startswith("dgk_")
    assert plaintext != digest
    assert verify_api_key(plaintext, digest)
    assert not verify_api_key(plaintext + "x", digest)
