import io
import pytest
import polars as pl

from dataprep.connectors.s3 import S3Connector

S3_CONFIG = {
    "aws_access_key_id": "minioadmin",
    "aws_secret_access_key": "minioadmin",
    "region": "us-east-1",
    "endpoint_url": "http://localhost:9000",
    "bucket": "testbucket",
    "default_url": "s3://testbucket/test_data.parquet",
}

BUCKET = "testbucket"
PARQUET_KEY = "test_data.parquet"
CSV_KEY = "test_data.csv"


@pytest.fixture
def connector():
    return S3Connector(S3_CONFIG)


async def _upload_parquet(connector: S3Connector, df: pl.DataFrame, key: str):
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    async with connector._client() as client:
        await client.put_object(Bucket=BUCKET, Key=key, Body=buf.read())


async def _upload_csv(connector: S3Connector, df: pl.DataFrame, key: str):
    buf = io.BytesIO()
    df.write_csv(buf)
    buf.seek(0)
    async with connector._client() as client:
        await client.put_object(Bucket=BUCKET, Key=key, Body=buf.read())


async def _delete_key(connector: S3Connector, key: str):
    async with connector._client() as client:
        try:
            await client.delete_object(Bucket=BUCKET, Key=key)
        except Exception:
            pass


@pytest.fixture
async def parquet_object(connector: S3Connector, sample_df: pl.DataFrame):
    await _upload_parquet(connector, sample_df, PARQUET_KEY)
    yield
    await _delete_key(connector, PARQUET_KEY)


@pytest.fixture
async def csv_object(connector: S3Connector, sample_df: pl.DataFrame):
    await _upload_csv(connector, sample_df, CSV_KEY)
    yield
    await _delete_key(connector, CSV_KEY)


@pytest.fixture
async def partitioned_objects(connector: S3Connector, sample_df: pl.DataFrame):
    mid = len(sample_df) // 2
    part1 = sample_df.head(mid)
    part2 = sample_df.tail(len(sample_df) - mid)
    await _upload_parquet(connector, part1, "partitioned/part_0.parquet")
    await _upload_parquet(connector, part2, "partitioned/part_1.parquet")
    yield part1, part2
    await _delete_key(connector, "partitioned/part_0.parquet")
    await _delete_key(connector, "partitioned/part_1.parquet")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_test_connection(connector: S3Connector):
    status = await connector.test_connection()
    assert status.ok
    assert status.latency_ms >= 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_parquet(connector: S3Connector, parquet_object, sample_df: pl.DataFrame):
    df = await connector.read({"url": f"s3://{BUCKET}/{PARQUET_KEY}"})
    assert len(df) == len(sample_df)
    assert set(df.columns) == set(sample_df.columns)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_csv(connector: S3Connector, csv_object, sample_df: pl.DataFrame):
    df = await connector.read({"url": f"s3://{BUCKET}/{CSV_KEY}"})
    assert len(df) == len(sample_df)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_stream_yields_one_frame_per_key(connector: S3Connector, parquet_object, sample_df: pl.DataFrame):
    frames = []
    async for chunk in connector.read_stream({"url": f"s3://{BUCKET}/{PARQUET_KEY}"}):
        frames.append(chunk)
    assert len(frames) == 1
    assert len(frames[0]) == len(sample_df)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_glob_pattern(connector: S3Connector, partitioned_objects, sample_df: pl.DataFrame):
    df = await connector.read({"url": f"s3://{BUCKET}/partitioned/part_*.parquet"})
    assert len(df) == len(sample_df)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_stream_glob_yields_per_file(connector: S3Connector, partitioned_objects):
    part1, part2 = partitioned_objects
    expected_sizes = sorted([len(part1), len(part2)])
    frames = []
    async for chunk in connector.read_stream(
        {"url": f"s3://{BUCKET}/partitioned/part_*.parquet"}
    ):
        frames.append(chunk)
    assert len(frames) == 2
    assert sorted(len(f) for f in frames) == expected_sizes


@pytest.mark.integration
@pytest.mark.asyncio
async def test_schema_parquet(connector: S3Connector, parquet_object, sample_df: pl.DataFrame):
    cfg = {**S3_CONFIG, "default_url": f"s3://{BUCKET}/{PARQUET_KEY}"}
    c = S3Connector(cfg)
    info = await c.schema()
    names = [col.name for col in info.columns]
    assert set(names) == set(sample_df.columns)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_parquet(connector: S3Connector, sample_df: pl.DataFrame):
    url = f"s3://{BUCKET}/output_test.parquet"
    result = await connector.write(sample_df, {"url": url})
    assert result.rows_written == len(sample_df)
    assert result.duration_s >= 0

    df = await connector.read({"url": url})
    assert len(df) == len(sample_df)
    await _delete_key(connector, "output_test.parquet")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_csv(connector: S3Connector, sample_df: pl.DataFrame):
    url = f"s3://{BUCKET}/output_test.csv"
    result = await connector.write(sample_df, {"url": url})
    assert result.rows_written == len(sample_df)

    df = await connector.read({"url": url})
    assert len(df) == len(sample_df)
    await _delete_key(connector, "output_test.csv")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connection_failure():
    bad = S3Connector({**S3_CONFIG, "endpoint_url": "http://localhost:19999"})
    status = await bad.test_connection()
    assert not status.ok
    assert status.error is not None