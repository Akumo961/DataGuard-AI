"""Defensive upload validation helpers.

Validation is deliberately independent from content extraction. Uploaded bytes must still be
processed in an isolated worker and malware-scanned by the deployment environment.
"""

import ipaddress
import socket
from pathlib import PurePath
from urllib.parse import urlparse

from dataguard.core.config import get_settings


class UploadRejected(ValueError):
    pass


def validate_filename(filename: str) -> str:
    name = PurePath(filename or "").name
    if not name or name in {".", ".."} or len(name) > 255 or name != filename:
        raise UploadRejected("Invalid filename")
    if any(ord(char) < 32 for char in name):
        raise UploadRejected("Invalid filename")
    return name


def validate_upload_metadata(*, filename: str, content_type: str, size_bytes: int) -> None:
    validate_filename(filename)
    settings = get_settings()
    if size_bytes < 0 or size_bytes > settings.max_upload_bytes:
        raise UploadRejected("Upload exceeds configured size limit")
    if content_type not in settings.upload_allowed_mime_types:
        raise UploadRejected("Unsupported content type")


def validate_outbound_url(url: str) -> None:
    """Reject non-HTTP(S), local, loopback and private destinations before egress."""
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise UploadRejected("Outbound URL is not allowed")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "metadata.google.internal"}:
        raise UploadRejected("Local/metadata destinations are not allowed")
    try:
        addresses = socket.getaddrinfo(
            host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM
        )
    except OSError as exc:
        raise UploadRejected("Unable to resolve outbound host") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise UploadRejected("Private or local outbound destination is not allowed")
