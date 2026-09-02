from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial

import boto3

from dataguard.connectors.base import Connector, ConnectorObject, ConnectorType


@dataclass(frozen=True)
class S3ConnectorConfig:
    bucket: str
    tenant_prefix: str
    endpoint_url: str | None = None
    region_name: str | None = None


class S3ObjectStorageConnector(Connector):
    """S3-compatible object storage adapter scoped to one tenant prefix.

    The IAM identity should have only s3:ListBucket on the configured prefix and
    s3:GetObject on that prefix. Delete/write permissions are intentionally absent.
    """

    type = ConnectorType.OBJECT_STORAGE

    def __init__(self, config: S3ConnectorConfig) -> None:
        prefix = config.tenant_prefix.strip("/")
        if not config.bucket or not prefix:
            raise ValueError("bucket and tenant_prefix are required")
        self.config = S3ConnectorConfig(
            bucket=config.bucket,
            tenant_prefix=f"{prefix}/",
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
        )
        self._client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
        )

    def _key(self, object_id: str) -> str:
        normalized = object_id.lstrip("/")
        if not normalized or normalized.startswith("../") or "/../" in normalized:
            raise ValueError("invalid object id")
        return f"{self.config.tenant_prefix}{normalized}"

    async def health(self) -> bool:
        try:
            await asyncio.to_thread(
                self._client.head_bucket,
                Bucket=self.config.bucket,
            )
            return True
        except Exception:
            return False

    async def list_objects(self, prefix: str | None = None) -> AsyncIterator[ConnectorObject]:
        requested = (prefix or "").lstrip("/")
        key_prefix = f"{self.config.tenant_prefix}{requested}"
        continuation: str | None = None
        while True:
            kwargs = {
                "Bucket": self.config.bucket,
                "Prefix": key_prefix,
                "MaxKeys": 1000,
            }
            if continuation:
                kwargs["ContinuationToken"] = continuation
            response = await asyncio.to_thread(self._client.list_objects_v2, **kwargs)
            for item in response.get("Contents", []):
                key = str(item["Key"])
                if not key.startswith(self.config.tenant_prefix):
                    continue
                yield ConnectorObject(
                    object_id=key.removeprefix(self.config.tenant_prefix),
                    name=key.rsplit("/", 1)[-1],
                    content_type=None,
                    size=item.get("Size"),
                    modified_at=item.get("LastModified").isoformat()
                    if item.get("LastModified")
                    else None,
                    source_uri=f"s3://{self.config.bucket}/{key}",
                )
            if not response.get("IsTruncated"):
                break
            continuation = response.get("NextContinuationToken")
            if not continuation:
                break

    async def read_object(self, object_id: str) -> AsyncIterator[bytes]:
        key = self._key(object_id)
        response = await asyncio.to_thread(
            self._client.get_object,
            Bucket=self.config.bucket,
            Key=key,
        )
        body = response["Body"]
        try:
            while True:
                chunk = await asyncio.to_thread(partial(body.read, 1024 * 1024))
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(body.close)
