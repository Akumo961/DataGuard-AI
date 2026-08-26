import pytest

from dataguard.security.uploads import UploadRejected, validate_filename, validate_outbound_url


def test_path_traversal_filename_is_rejected() -> None:
    with pytest.raises(UploadRejected):
        validate_filename("../../secret.txt")


def test_localhost_egress_is_rejected() -> None:
    with pytest.raises(UploadRejected):
        validate_outbound_url("http://localhost:8080/admin")


def test_credentials_in_url_are_rejected() -> None:
    with pytest.raises(UploadRejected):
        validate_outbound_url("https://user:pass@example.com/")
