import io
import time
import asyncio
import subprocess
from pathlib import Path
from typing import AsyncGenerator

import aioboto3
import polars as pl
import pytest
import pytest_asyncio

from dataprep.connectors.s3 import S3Connector

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"

MINIO_ENDPOINT = "http://localhost:9000"
BUCKET = "testbucket"

S3_CONFIG = {
    "aws_access_key_id": "minioadmin",
    "aws_secret_access_key": "minioadmin",
    "endpoint_url": MINIO_ENDPOINT,
    "region": "us-east-1",
    "bucket": BUCKET,
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: mark test as requiring Docker/MinIO"
    )


def _wait_for_minio(timeout: float = 30) -> None:
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{MINIO_ENDPOINT}/minio/health/live", timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("MinIO did not become ready in time")


def _wait_for_bucket(timeout: float = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        loop = asyncio.new_event_loop()
        try:
            async def _check():
                session = aioboto3.Session()
                async with session.client(
                    "s3",
                    endpoint_url=MINIO_ENDPOINT,
                    aws_access_key_id="minioadmin",
                    aws_secret_access_key="minioadmin",
                    region_name="us-east-1",
                ) as client:
                    await client.head_bucket(Bucket=BUCKET)

            loop.run_until_complete(_check())
            return
        except Exception:
            time.sleep(0.5)
        finally:
            loop.close()
    raise RuntimeError("MinIO bucket 'testbucket' was not created in time")


def pytest_sessionstart(session: pytest.Session) -> None:
    result = subprocess.run(
        ["docker", "compose", "-f", "docker-compose.test.yml", "up", "-d", "--wait"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start Docker services:\n{result.stderr}")

    _wait_for_minio(timeout=30)
    _wait_for_bucket(timeout=30)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.test.yml", "down", "-v"],
        capture_output=True,
    )


@pytest.fixture(scope="session")
def sample_df() -> pl.DataFrame:
    return pl.read_csv(DATA_CSV, try_parse_dates=True)


@pytest.fixture(scope="session")
def s3_connector() -> S3Connector:
    return S3Connector(S3_CONFIG)


def _csv_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_csv(buf)
    return buf.getvalue()


def _parquet_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


async def _put(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        region_name="us-east-1",
    ) as client:
        await client.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)


async def _delete(*keys: str) -> None:
    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        region_name="us-east-1",
    ) as client:
        for key in keys:
            await client.delete_object(Bucket=BUCKET, Key=key)


async def _list_keys(prefix: str) -> list[str]:
    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        region_name="us-east-1",
    ) as client:
        resp = await client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]


async def _get_bytes(key: str) -> bytes:
    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        region_name="us-east-1",
    ) as client:
        resp = await client.get_object(Bucket=BUCKET, Key=key)
        return await resp["Body"].read()


@pytest_asyncio.fixture(autouse=True)
async def _clean_bucket() -> AsyncGenerator[None, None]:
    yield
    keys = await _list_keys("")
    if keys:
        await _delete(*keys)


@pytest.mark.integration
class TestS3ReadIntegration:
    @pytest.mark.asyncio
    async def test_read_csv_returns_correct_shape(self, s3_connector, sample_df):
        key = "read/data.csv"
        await _put(key, _csv_bytes(sample_df), "text/csv")

        result = await s3_connector.read({"url": f"s3://{BUCKET}/{key}"})

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_df)
        assert set(result.columns) == set(sample_df.columns)

    @pytest.mark.asyncio
    async def test_read_parquet_returns_correct_shape(self, s3_connector, sample_df):
        key = "read/data.parquet"
        await _put(key, _parquet_bytes(sample_df))

        result = await s3_connector.read({"url": f"s3://{BUCKET}/{key}"})

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_df)
        assert set(result.columns) == set(sample_df.columns)

    @pytest.mark.asyncio
    async def test_read_csv_values_match_source(self, s3_connector, sample_df):
        key = "read/values.csv"
        await _put(key, _csv_bytes(sample_df), "text/csv")

        result = await s3_connector.read({"url": f"s3://{BUCKET}/{key}"})

        assert result.sort(result.columns[0]).equals(
            sample_df.sort(sample_df.columns[0])
        )

    @pytest.mark.asyncio
    async def test_read_glob_csv_concatenates_all_files(self, s3_connector, sample_df):
        half = len(sample_df) // 2
        part1 = sample_df[:half]
        part2 = sample_df[half:]
        await _put("glob/part1.csv", _csv_bytes(part1), "text/csv")
        await _put("glob/part2.csv", _csv_bytes(part2), "text/csv")

        result = await s3_connector.read({"url": f"s3://{BUCKET}/glob/*.csv"})

        assert len(result) == len(sample_df)

    @pytest.mark.asyncio
    async def test_read_glob_parquet_concatenates_all_files(self, s3_connector, sample_df):
        half = len(sample_df) // 2
        await _put("pglob/a.parquet", _parquet_bytes(sample_df[:half]))
        await _put("pglob/b.parquet", _parquet_bytes(sample_df[half:]))

        result = await s3_connector.read({"url": f"s3://{BUCKET}/pglob/*.parquet"})

        assert len(result) == len(sample_df)

    @pytest.mark.asyncio
    async def test_read_glob_empty_prefix_returns_empty(self, s3_connector):
        result = await s3_connector.read({"url": f"s3://{BUCKET}/nonexistent/*.csv"})

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0


@pytest.mark.integration
class TestS3WriteIntegration:
    @pytest.mark.asyncio
    async def test_write_csv_object_exists_in_bucket(self, s3_connector, sample_df):
        key = "write/out.csv"
        result = await s3_connector.write(sample_df, {"url": f"s3://{BUCKET}/{key}"})

        assert result.rows_written == len(sample_df)
        keys = await _list_keys("write/")
        assert key in keys

    @pytest.mark.asyncio
    async def test_write_parquet_object_exists_in_bucket(self, s3_connector, sample_df):
        key = "write/out.parquet"
        result = await s3_connector.write(sample_df, {"url": f"s3://{BUCKET}/{key}"})

        assert result.rows_written == len(sample_df)
        keys = await _list_keys("write/")
        assert key in keys

    @pytest.mark.asyncio
    async def test_write_csv_content_is_readable(self, s3_connector, sample_df):
        key = "write/readable.csv"
        await s3_connector.write(sample_df, {"url": f"s3://{BUCKET}/{key}"})

        raw = await _get_bytes(key)
        recovered = pl.read_csv(io.BytesIO(raw))
        assert len(recovered) == len(sample_df)
        assert set(recovered.columns) == set(sample_df.columns)

    @pytest.mark.asyncio
    async def test_write_parquet_content_is_readable(self, s3_connector, sample_df):
        key = "write/readable.parquet"
        await s3_connector.write(sample_df, {"url": f"s3://{BUCKET}/{key}"})

        raw = await _get_bytes(key)
        recovered = pl.read_parquet(io.BytesIO(raw))
        assert len(recovered) == len(sample_df)

    @pytest.mark.asyncio
    async def test_write_returns_non_negative_duration(self, s3_connector, sample_df):
        result = await s3_connector.write(
            sample_df, {"url": f"s3://{BUCKET}/write/timing.parquet"}
        )
        assert result.duration_s >= 0

    @pytest.mark.asyncio
    async def test_write_overwrites_existing_object(self, s3_connector, sample_df):
        key = "write/overwrite.csv"
        small_df = sample_df[:1]
        await s3_connector.write(small_df, {"url": f"s3://{BUCKET}/{key}"})
        await s3_connector.write(sample_df, {"url": f"s3://{BUCKET}/{key}"})

        raw = await _get_bytes(key)
        recovered = pl.read_csv(io.BytesIO(raw))
        assert len(recovered) == len(sample_df)


@pytest.mark.integration
class TestS3StreamIntegration:
    @pytest.mark.asyncio
    async def test_stream_yields_one_chunk_per_file(self, s3_connector, sample_df):
        await _put("stream/a.parquet", _parquet_bytes(sample_df))
        await _put("stream/b.parquet", _parquet_bytes(sample_df))

        chunks = []
        async for chunk in s3_connector.read_stream(
            {"url": f"s3://{BUCKET}/stream/*.parquet"}
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert all(isinstance(c, pl.DataFrame) for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_total_rows_match_source(self, s3_connector, sample_df):
        half = len(sample_df) // 2
        await _put("srows/p1.parquet", _parquet_bytes(sample_df[:half]))
        await _put("srows/p2.parquet", _parquet_bytes(sample_df[half:]))

        total = 0
        async for chunk in s3_connector.read_stream(
            {"url": f"s3://{BUCKET}/srows/*.parquet"}
        ):
            total += len(chunk)

        assert total == len(sample_df)

    @pytest.mark.asyncio
    async def test_stream_empty_prefix_yields_nothing(self, s3_connector):
        chunks = []
        async for chunk in s3_connector.read_stream(
            {"url": f"s3://{BUCKET}/empty/*.parquet"}
        ):
            chunks.append(chunk)

        assert chunks == []


@pytest.mark.integration
class TestS3ConnectionIntegration:
    @pytest.mark.asyncio
    async def test_test_connection_returns_ok(self, s3_connector):
        status = await s3_connector.test_connection()

        assert status.ok is True
        assert status.latency_ms is not None
        assert status.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_test_connection_wrong_bucket_returns_not_ok(self):
        bad_connector = S3Connector({
            **S3_CONFIG,
            "bucket": "nonexistent-bucket-xyz",
        })
        status = await bad_connector.test_connection()

        assert status.ok is False
        assert status.error is not None

    @pytest.mark.asyncio
    async def test_test_connection_wrong_credentials_returns_not_ok(self):
        bad_connector = S3Connector({
            **S3_CONFIG,
            "aws_access_key_id": "wrong",
            "aws_secret_access_key": "wrong",
        })
        status = await bad_connector.test_connection()

        assert status.ok is False
        assert status.error is not None

    @pytest.mark.asyncio
    async def test_test_connection_latency_is_reasonable(self, s3_connector):
        status = await s3_connector.test_connection()

        assert status.ok is True
        assert status.latency_ms < 5000


@pytest.mark.integration
class TestS3SchemaIntegration:
    @pytest.mark.asyncio
    async def test_schema_returns_correct_column_count(self, sample_df):
        key = "schema/data.parquet"
        await _put(key, _parquet_bytes(sample_df))
        connector = S3Connector({**S3_CONFIG, "default_url": f"s3://{BUCKET}/{key}"})

        schema = await connector.schema()

        assert len(schema.columns) == len(sample_df.columns)

    @pytest.mark.asyncio
    async def test_schema_column_names_match(self, sample_df):
        key = "schema/named.parquet"
        await _put(key, _parquet_bytes(sample_df))
        connector = S3Connector({**S3_CONFIG, "default_url": f"s3://{BUCKET}/{key}"})

        schema = await connector.schema()

        schema_names = {c.name for c in schema.columns}
        assert schema_names == set(sample_df.columns)

    @pytest.mark.asyncio
    async def test_schema_without_default_url_returns_empty(self):
        connector = S3Connector(S3_CONFIG)
        schema = await connector.schema()

        assert schema.columns == []

    @pytest.mark.asyncio
    async def test_schema_columns_all_nullable(self, sample_df):
        key = "schema/nullable.parquet"
        await _put(key, _parquet_bytes(sample_df))
        connector = S3Connector({**S3_CONFIG, "default_url": f"s3://{BUCKET}/{key}"})

        schema = await connector.schema()

        assert all(c.nullable for c in schema.columns)