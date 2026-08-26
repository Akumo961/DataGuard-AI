from __future__ import annotations

import io
import zipfile


class UnsafeArchiveError(ValueError):
    pass


def validate_ooxml_archive(content: bytes, *, max_members: int = 500, max_uncompressed_bytes: int = 50 * 1024 * 1024) -> None:
    """Reject suspicious OOXML ZIP containers before handing them to parsers."""
    try:
        with zipfile.ZipFile(io.BytesIO(content), "r") as archive:
            members = archive.infolist()
            if len(members) > max_members:
                raise UnsafeArchiveError("archive contains too many members")
            total = 0
            for member in members:
                name = member.filename.replace("\\", "/")
                if name.startswith("/") or "../" in name.split("/"):
                    raise UnsafeArchiveError("archive contains an unsafe path")
                if member.file_size < 0:
                    raise UnsafeArchiveError("invalid archive member size")
                total += member.file_size
                if total > max_uncompressed_bytes:
                    raise UnsafeArchiveError("archive expands beyond the configured limit")
    except zipfile.BadZipFile as exc:
        raise UnsafeArchiveError("invalid OOXML archive") from exc
