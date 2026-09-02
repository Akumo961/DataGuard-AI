from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dataguard.connectors.s3 import S3ConnectorConfig, S3ObjectStorageConnector


class FakeBody:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = iter(chunks)

    def read(self, _size: int) -> bytes:
        return next(self.chunks, b"")

    def close(self) -> None:
        return None


class FakeS3:
    def __init__(self) -> None:
        self.requested_keys: list[str] = []

    def head_bucket(self, **kwargs):
        assert kwargs == {"Bucket": "tenant-bucket"}

    def list_objects_v2(self, **kwargs):
        assert kwargs["Prefix"] == "tenant-a/"
        return {
            "Contents": [
                {
                    "Key": "tenant-a/report.pdf",
                    "Size": 12,
                    "LastModified": datetime(2026, 9, 1, tzinfo=timezone.utc),
                }
            ],
            "IsTruncated": False,
        }

    def get_object(self, **kwargs):
        self.requested_keys.append(kwargs["Key"])
        return {"Body": FakeBody([b"hello", b" world"])}


@pytest.mark.asyncio
async def test_s3_connector_scopes_reads_to_tenant_prefix(monkeypatch) -> None:
    fake = FakeS3()
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: fake)
    connector = S3ObjectStorageConnector(
        S3ConnectorConfig(bucket="tenant-bucket", tenant_prefix="tenant-a")
    )

    assert await connector.health() is True
    objects = [item async for item in connector.list_objects()]
    assert objects[0].object_id == "report.pdf"
    assert objects[0].source_uri == "s3://tenant-bucket/tenant-a/report.pdf"

    chunks = [chunk async for chunk in connector.read_object("report.pdf")]
    assert b"".join(chunks) == b"hello world"
    assert fake.requested_keys == ["tenant-a/report.pdf"]


@pytest.mark.asyncio
async def test_s3_connector_rejects_path_escape(monkeypatch) -> None:
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: FakeS3())
    connector = S3ObjectStorageConnector(
        S3ConnectorConfig(bucket="tenant-bucket", tenant_prefix="tenant-a")
    )
    with pytest.raises(ValueError):
        await anext(connector.read_object("../tenant-b/report.pdf"))
