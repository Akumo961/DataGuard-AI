import io
import zipfile

import pytest

from dataguard.processing.archive import UnsafeArchiveError, validate_ooxml_archive


def test_valid_small_archive() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", "<document/>")
    validate_ooxml_archive(buffer.getvalue())


def test_rejects_unsafe_archive_path() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../escape.txt", "x")
    with pytest.raises(UnsafeArchiveError):
        validate_ooxml_archive(buffer.getvalue())


def test_rejects_too_many_members() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index in range(3):
            archive.writestr(f"file{index}.txt", "x")
    with pytest.raises(UnsafeArchiveError):
        validate_ooxml_archive(buffer.getvalue(), max_members=2)
