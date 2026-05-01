from dataclasses import dataclass
from typing import Optional, AsyncGenerator, Any
import io

import polars as pl
import pyarrow.parquet as pq
import aioboto3

from .base import BaseConnector, ConnectionStatus
from ..utils.connection import timed_connect
from ..utils.streaming import estimate_batch_size
from ..utils.file_process import parse_bytes, get_format, resolve_key


@dataclass
class S3ConnectorConfig:
    bucket: str
    region: str = "us-east-1"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    prefix: str = ""
    read_only: bool = False
    format_filter: Optional[list[str]] = None
    encoding: str = "utf-8"


class S3Connector(BaseConnector[S3ConnectorConfig]):
    def __init__(self, config: S3ConnectorConfig):
        super().__init__(config)
        self.session = aioboto3.Session(
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name=config.region,
        )

    def _client(self):
        return self.session.client("s3", endpoint_url=self.config.endpoint_url)

    async def connect(self) -> ConnectionStatus:
        async def probe():
            async with self._client() as s3:
                await s3.head_bucket(Bucket=self.config.bucket)

        ok, latency_ms, error = await timed_connect(probe)
        return ConnectionStatus(ok=ok, latency_ms=latency_ms, error=error)

    async def cleanup(self) -> None:
        return

    async def read(
        self,
        path: str,
        format: Optional[str] = None,
        stream: bool = False,
        batch_size: Optional[int] = None,
    ) -> "pl.DataFrame | AsyncGenerator[pl.DataFrame, Any]":
        key = resolve_key(self.config.prefix, path)
        fmt = get_format(path, format)

        if self.config.format_filter and fmt not in self.config.format_filter:
            raise ValueError("format not allowed")

        if not stream:
            async with self._client() as s3:
                obj = await s3.get_object(Bucket=self.config.bucket, Key=key)
                data = await obj["Body"].read()
            return parse_bytes(data, fmt, self.config.encoding)

        async def generate() -> AsyncGenerator[pl.DataFrame, Any]:
            if fmt == "parquet":
                async with self._client() as s3:
                    obj = await s3.get_object(Bucket=self.config.bucket, Key=key)
                    data = await obj["Body"].read()
                for batch in pq.ParquetFile(io.BytesIO(data)).iter_batches():
                    yield pl.from_arrow(batch)
            else:
                async with self._client() as s3:
                    obj = await s3.get_object(Bucket=self.config.bucket, Key=key)
                    data = await obj["Body"].read()
                df = parse_bytes(data, fmt, self.config.encoding)
                resolved = batch_size or estimate_batch_size(len(data), len(df))
                for i in range(0, len(df), resolved):
                    yield df.slice(i, resolved)

        return generate()

    async def write(
        self,
        df: pl.DataFrame,
        path: str,
        format: Optional[str] = None,
    ) -> None:
        if self.config.read_only:
            raise PermissionError("read only")

        key = resolve_key(self.config.prefix, path)
        fmt = get_format(path, format)
        buf = io.BytesIO()

        if fmt == "parquet":
            df.write_parquet(buf)
        elif fmt == "csv":
            df.write_csv(buf)
        elif fmt == "json":
            df.write_json(buf)
        else:
            raise ValueError(f"unsupported format: {fmt}")

        buf.seek(0)
        async with self._client() as s3:
            await s3.put_object(Bucket=self.config.bucket, Key=key, Body=buf.read())

    async def list_objects(self, prefix: Optional[str] = None) -> list[str]:
        base = resolve_key(self.config.prefix, prefix or "")
        keys = []
        async with self._client() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self.config.bucket, Prefix=base):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        return keys