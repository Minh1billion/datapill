import asyncio
import fnmatch
import io
import random
import time
from typing import AsyncGenerator, Any
from urllib.parse import urlparse

import aioboto3
import polars as pl
import pyarrow.parquet as pq
import pyarrow as pa

from .base import BaseConnector, ColumnMeta, ConnectionStatus, SchemaInfo, WriteResult


_MAX_RETRIES = 3
_BASE_BACKOFF = 0.5
_RETRYABLE_CODES = {"RequestTimeout", "ServiceUnavailable", "SlowDown", "InternalError"}


def _parse_s3_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def _is_glob(path: str) -> bool:
    return any(c in path for c in ("*", "?", "["))


class S3Connector(BaseConnector):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._session = aioboto3.Session(
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key"),
            region_name=config.get("region", "us-east-1"),
        )
        self._client_kwargs: dict[str, Any] = {}
        if endpoint := config.get("endpoint_url"):
            self._client_kwargs["endpoint_url"] = endpoint

    def _client(self):
        return self._session.client("s3", **self._client_kwargs)

    async def _list_keys(self, client, bucket: str, prefix: str, pattern: str) -> list[str]:
        paginator = client.get_paginator("list_objects_v2")
        keys = []
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if fnmatch.fnmatch(key, pattern):
                    keys.append(key)
        return sorted(keys)

    async def _read_key(self, client, bucket: str, key: str, fmt: str) -> pl.DataFrame:
        resp = await client.get_object(Bucket=bucket, Key=key)
        body = await resp["Body"].read()
        if fmt == "parquet":
            table = pq.read_table(io.BytesIO(body))
            return pl.from_arrow(table)
        elif fmt == "csv":
            return pl.read_csv(io.BytesIO(body))
        raise ValueError(f"Unsupported format: {fmt}")

    def _detect_format(self, key: str, options: dict) -> str:
        if fmt := options.get("format"):
            return fmt
        if key.endswith(".parquet"):
            return "parquet"
        if key.endswith(".csv"):
            return "csv"
        return "csv"

    async def _resolve_keys(self, client, url: str) -> tuple[str, list[str], str]:
        bucket, path = _parse_s3_url(url)
        if _is_glob(path):
            parts = path.split("/")
            glob_idx = next(i for i, p in enumerate(parts) if _is_glob(p))
            prefix = "/".join(parts[:glob_idx]) + ("/" if glob_idx > 0 else "")
            pattern = path
            keys = await self._list_keys(client, bucket, prefix, pattern)
        else:
            keys = [path]
        return bucket, keys, path

    async def read(
        self, query: dict[str, Any], options: dict[str, Any] | None = None
    ) -> pl.DataFrame:
        opts = options or {}
        url = query["url"]

        async with self._client() as client:
            bucket, keys, path = await self._resolve_keys(client, url)
            fmt = self._detect_format(keys[0] if keys else path, opts)

            frames = []
            for key in keys:
                df = await self._read_key(client, bucket, key, fmt)
                frames.append(df)

        if not frames:
            return pl.DataFrame()
        return pl.concat(frames)

    async def read_stream(
        self, query: dict[str, Any], options: dict[str, Any] | None = None
    ) -> AsyncGenerator[pl.DataFrame, None]:
        opts = options or {}
        url = query["url"]

        async with self._client() as client:
            bucket, keys, path = await self._resolve_keys(client, url)
            fmt = self._detect_format(keys[0] if keys else path, opts)

            for key in keys:
                df = await self._read_key(client, bucket, key, fmt)
                yield df

    async def schema(self) -> SchemaInfo:
        url = self.config.get("default_url")
        if not url:
            return SchemaInfo(columns=[])

        async with self._client() as client:
            bucket, path = _parse_s3_url(url)
            resp = await client.get_object(Bucket=bucket, Key=path)
            body = await resp["Body"].read()

        fmt = self._detect_format(path, {})
        if fmt == "parquet":
            schema = pq.read_schema(io.BytesIO(body))
            columns = [
                ColumnMeta(
                    name=field.name,
                    dtype=str(field.type),
                    nullable=field.nullable,
                )
                for field in schema
            ]
        else:
            df = pl.read_csv(io.BytesIO(body), n_rows=0)
            columns = [
                ColumnMeta(name=col, dtype=str(dtype), nullable=True)
                for col, dtype in zip(df.columns, df.dtypes)
            ]
        return SchemaInfo(columns=columns)

    async def test_connection(self) -> ConnectionStatus:
        t0 = time.perf_counter()
        try:
            async with self._client() as client:
                bucket = self.config.get("bucket")
                if bucket:
                    await client.head_bucket(Bucket=bucket)
                else:
                    await client.list_buckets()
            return ConnectionStatus(ok=True, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return ConnectionStatus(ok=False, error=str(exc))

    async def write(
        self,
        df: pl.DataFrame,
        target: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> WriteResult:
        opts = options or {}
        url = target["url"]
        bucket, key = _parse_s3_url(url)
        fmt = self._detect_format(key, opts)

        t0 = time.perf_counter()
        buf = io.BytesIO()

        if fmt == "parquet":
            df.write_parquet(buf)
        else:
            df.write_csv(buf)

        buf.seek(0)
        async with self._client() as client:
            await client.put_object(Bucket=bucket, Key=key, Body=buf.read())

        return WriteResult(rows_written=len(df), duration_s=time.perf_counter() - t0)

    async def close(self) -> None:
        pass