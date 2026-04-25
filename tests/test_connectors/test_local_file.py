import pytest
import polars as pl
from pathlib import Path

from dataprep.connectors.local_file import LocalFileConnector

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def connector():
    return LocalFileConnector()


@pytest.fixture
def csv_path():
    return FIXTURES_DIR / "data.csv"


@pytest.fixture
def tmp_parquet(tmp_path, sample_df):
    p = tmp_path / "data.parquet"
    sample_df.write_parquet(p)
    return p


@pytest.fixture
def tmp_csv(tmp_path, sample_df):
    p = tmp_path / "data.csv"
    sample_df.write_csv(p)
    return p


@pytest.fixture
def tmp_ndjson(tmp_path, sample_df):
    p = tmp_path / "data.ndjson"
    sample_df.write_ndjson(p)
    return p


@pytest.fixture
def tmp_xlsx(tmp_path, sample_df):
    p = tmp_path / "data.xlsx"
    sample_df.write_excel(p)
    return p


@pytest.mark.asyncio
async def test_read_csv(connector: LocalFileConnector, csv_path: Path, sample_df: pl.DataFrame):
    df = await connector.read({"path": str(csv_path)})
    assert len(df) == len(sample_df)
    assert set(df.columns) == set(sample_df.columns)


@pytest.mark.asyncio
async def test_read_parquet(connector: LocalFileConnector, tmp_parquet: Path, sample_df: pl.DataFrame):
    df = await connector.read({"path": str(tmp_parquet)})
    assert len(df) == len(sample_df)


@pytest.mark.asyncio
async def test_read_ndjson(connector: LocalFileConnector, tmp_ndjson: Path, sample_df: pl.DataFrame):
    df = await connector.read({"path": str(tmp_ndjson)})
    assert len(df) == len(sample_df)


@pytest.mark.asyncio
async def test_read_xlsx(connector: LocalFileConnector, tmp_xlsx: Path, sample_df: pl.DataFrame):
    df = await connector.read({"path": str(tmp_xlsx)})
    assert len(df) == len(sample_df)


@pytest.mark.asyncio
async def test_read_csv_n_rows(connector: LocalFileConnector, csv_path: Path, sample_df: pl.DataFrame):
    n = min(3, len(sample_df))
    df = await connector.read({"path": str(csv_path)}, {"n_rows": n})
    assert len(df) == n


@pytest.mark.asyncio
async def test_read_stream_total_rows(connector: LocalFileConnector, csv_path: Path, sample_df: pl.DataFrame):
    frames = []
    async for chunk in connector.read_stream({"path": str(csv_path)}, {"batch_size": 3}):
        frames.append(chunk)
    assert sum(len(f) for f in frames) == len(sample_df)


@pytest.mark.asyncio
async def test_read_stream_batch_size_respected(connector: LocalFileConnector, csv_path: Path, sample_df: pl.DataFrame):
    batch_size = max(1, len(sample_df) // 3)
    frames = []
    async for chunk in connector.read_stream({"path": str(csv_path)}, {"batch_size": batch_size}):
        frames.append(chunk)
    assert all(len(f) <= batch_size for f in frames)
    assert len(frames) > 1


@pytest.mark.asyncio
async def test_read_stream_adaptive_batch(connector: LocalFileConnector, tmp_parquet: Path, sample_df: pl.DataFrame):
    frames = []
    async for chunk in connector.read_stream({"path": str(tmp_parquet)}):
        frames.append(chunk)
    assert sum(len(f) for f in frames) == len(sample_df)


@pytest.mark.asyncio
async def test_schema_csv(csv_path: Path, sample_df: pl.DataFrame):
    c = LocalFileConnector({"default_path": str(csv_path)})
    info = await c.schema()
    names = [col.name for col in info.columns]
    assert set(names) == set(sample_df.columns)


@pytest.mark.asyncio
async def test_schema_parquet(tmp_parquet: Path, sample_df: pl.DataFrame):
    c = LocalFileConnector({"default_path": str(tmp_parquet)})
    info = await c.schema()
    assert len(info.columns) == len(sample_df.columns)


@pytest.mark.asyncio
async def test_schema_missing_path():
    c = LocalFileConnector({"default_path": "/nonexistent/path/file.csv"})
    info = await c.schema()
    assert info.columns == []


@pytest.mark.asyncio
async def test_test_connection_existing_file(csv_path: Path):
    c = LocalFileConnector({"default_path": str(csv_path)})
    status = await c.test_connection()
    assert status.ok


@pytest.mark.asyncio
async def test_test_connection_missing_file():
    c = LocalFileConnector({"default_path": "/does/not/exist.csv"})
    status = await c.test_connection()
    assert not status.ok


@pytest.mark.asyncio
async def test_write_csv(connector: LocalFileConnector, tmp_path: Path, sample_df: pl.DataFrame):
    out = tmp_path / "out.csv"
    result = await connector.write(sample_df, {"path": str(out)})
    assert result.rows_written == len(sample_df)
    assert out.exists()
    df = pl.read_csv(out)
    assert len(df) == len(sample_df)


@pytest.mark.asyncio
async def test_write_parquet(connector: LocalFileConnector, tmp_path: Path, sample_df: pl.DataFrame):
    out = tmp_path / "out.parquet"
    result = await connector.write(sample_df, {"path": str(out)})
    assert result.rows_written == len(sample_df)
    df = pl.read_parquet(out)
    assert len(df) == len(sample_df)


@pytest.mark.asyncio
async def test_write_ndjson(connector: LocalFileConnector, tmp_path: Path, sample_df: pl.DataFrame):
    out = tmp_path / "out.ndjson"
    result = await connector.write(sample_df, {"path": str(out)})
    assert result.rows_written == len(sample_df)
    df = pl.read_ndjson(out)
    assert len(df) == len(sample_df)


@pytest.mark.asyncio
async def test_write_xlsx(connector: LocalFileConnector, tmp_path: Path, sample_df: pl.DataFrame):
    out = tmp_path / "out.xlsx"
    result = await connector.write(sample_df, {"path": str(out)})
    assert result.rows_written == len(sample_df)
    df = pl.read_excel(out)
    assert len(df) == len(sample_df)


@pytest.mark.asyncio
async def test_write_returns_duration(connector: LocalFileConnector, tmp_path: Path, sample_df: pl.DataFrame):
    out = tmp_path / "out.csv"
    result = await connector.write(sample_df, {"path": str(out)})
    assert result.duration_s >= 0


@pytest.mark.asyncio
async def test_write_unsupported_format_raises(connector: LocalFileConnector, tmp_path: Path, sample_df: pl.DataFrame):
    out = tmp_path / "out.avro"
    with pytest.raises(ValueError, match="Unsupported"):
        await connector.write(sample_df, {"path": str(out)})


@pytest.mark.asyncio
async def test_read_unsupported_format_raises(connector: LocalFileConnector, tmp_path: Path):
    out = tmp_path / "file.avro"
    out.write_text("data")
    with pytest.raises(ValueError, match="Unsupported"):
        await connector.read({"path": str(out)})


@pytest.mark.asyncio
async def test_write_csv_custom_delimiter(connector: LocalFileConnector, tmp_path: Path, sample_df: pl.DataFrame):
    out = tmp_path / "out.csv"
    await connector.write(sample_df, {"path": str(out)}, {"delimiter": ";"})
    df = pl.read_csv(out, separator=";")
    assert len(df) == len(sample_df)