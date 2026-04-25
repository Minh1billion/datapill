from pathlib import Path

import polars as pl
import pytest

from dataprep.connectors.local_file import LocalFileConnector, _adaptive_batch_size


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"
DATA_XLSX = FIXTURES_DIR / "data.xlsx"


class TestAdaptiveBatchSize:
    def test_hint_within_bounds(self, sample_df):
        assert _adaptive_batch_size(sample_df, 50_000) == 50_000

    def test_hint_below_min_clamps_to_minimum(self, sample_df):
        assert _adaptive_batch_size(sample_df, 1) == 10_000

    def test_hint_above_max_clamps_to_maximum(self, sample_df):
        assert _adaptive_batch_size(sample_df, 999_999_999) == 500_000

    def test_no_hint_returns_value_within_bounds(self, sample_df):
        result = _adaptive_batch_size(sample_df, None)
        assert 10_000 <= result <= 500_000

    def test_empty_dataframe_returns_minimum(self):
        empty = pl.DataFrame({"a": pl.Series([], dtype=pl.Int32)})
        assert _adaptive_batch_size(empty, None) == 10_000


class TestLocalFileConnectorRead:
    @pytest.mark.asyncio
    async def test_read_csv(self, local_connector, sample_df):
        result = await local_connector.read({"path": str(DATA_CSV)})
        assert isinstance(result, pl.DataFrame)
        assert result.columns == sample_df.columns
        assert len(result) == len(sample_df)

    @pytest.mark.asyncio
    async def test_read_parquet(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "test.parquet"
        sample_df.write_parquet(path)

        result = await local_connector.read({"path": str(path)})
        assert result.shape == sample_df.shape

    @pytest.mark.asyncio
    async def test_read_ndjson(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "test.ndjson"
        sample_df.write_ndjson(path)

        result = await local_connector.read({"path": str(path)})
        assert len(result) == len(sample_df)

    @pytest.mark.asyncio
    async def test_read_xlsx(self, local_connector):
        result = await local_connector.read({"path": str(DATA_XLSX)})
        assert len(result) == 1000

    @pytest.mark.asyncio
    async def test_read_csv_with_custom_delimiter(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "semicolon.csv"
        sample_df.write_csv(path, separator=";")

        result = await local_connector.read({"path": str(path)}, {"delimiter": ";"})
        assert len(result) == len(sample_df)


class TestLocalFileConnectorStream:
    @pytest.mark.asyncio
    async def test_stream_yields_all_rows(self, local_connector, sample_df):
        total = 0
        async for chunk in local_connector.read_stream({"path": str(DATA_CSV)}):
            assert isinstance(chunk, pl.DataFrame)
            total += len(chunk)
        assert total == len(sample_df)

    @pytest.mark.asyncio
    async def test_stream_respects_batch_size(self, local_connector, sample_df):
        batch_size = max(1, len(sample_df) // 3)
        chunks = []
        async for chunk in local_connector.read_stream(
            {"path": str(DATA_CSV)}, {"batch_size": batch_size}
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1
        # All chunks except the last must be exactly batch_size
        for chunk in chunks[:-1]:
            assert len(chunk) == batch_size

    @pytest.mark.asyncio
    async def test_stream_schema_is_consistent_across_chunks(self, local_connector):
        schemas = []
        async for chunk in local_connector.read_stream({"path": str(DATA_CSV)}):
            schemas.append(chunk.schema)
        assert all(s == schemas[0] for s in schemas)


class TestLocalFileConnectorWrite:
    @pytest.mark.asyncio
    async def test_write_csv(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "out.csv"
        result = await local_connector.write(sample_df, {"path": str(path)})

        assert path.exists()
        assert result.rows_written == len(sample_df)
        assert len(pl.read_csv(path)) == len(sample_df)

    @pytest.mark.asyncio
    async def test_write_parquet(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "out.parquet"
        result = await local_connector.write(sample_df, {"path": str(path)})

        assert path.exists()
        assert result.rows_written == len(sample_df)

    @pytest.mark.asyncio
    async def test_write_ndjson(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "out.ndjson"
        result = await local_connector.write(sample_df, {"path": str(path)})

        assert path.exists()
        assert result.rows_written == len(sample_df)

    @pytest.mark.asyncio
    async def test_write_xlsx(self, local_connector, tmp_path):
        df = await local_connector.read({"path": str(DATA_XLSX)})
        path = tmp_path / "out.xlsx"
        result = await local_connector.write(df, {"path": str(path)})

        assert path.exists()
        assert result.rows_written == 1000

    @pytest.mark.asyncio
    async def test_write_unsupported_format_raises_value_error(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "out.txt"
        with pytest.raises(ValueError, match="Unsupported"):
            await local_connector.write(sample_df, {"path": str(path)})

    @pytest.mark.asyncio
    async def test_write_result_duration_is_non_negative(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "out.parquet"
        result = await local_connector.write(sample_df, {"path": str(path)})
        assert result.duration_s >= 0


class TestLocalFileConnectorSchema:
    @pytest.mark.asyncio
    async def test_schema_csv_returns_all_columns(self):
        connector = LocalFileConnector({"default_path": str(DATA_CSV)})
        schema = await connector.schema()

        assert len(schema.columns) > 0
        for col in schema.columns:
            assert col.name
            assert col.dtype

    @pytest.mark.asyncio
    async def test_schema_parquet_column_count_matches(self, sample_df, tmp_path):
        path = tmp_path / "test.parquet"
        sample_df.write_parquet(path)

        connector = LocalFileConnector({"default_path": str(path)})
        schema = await connector.schema()
        assert len(schema.columns) == len(sample_df.columns)

    @pytest.mark.asyncio
    async def test_schema_missing_file_returns_empty(self):
        connector = LocalFileConnector({"default_path": "/nonexistent/path.csv"})
        schema = await connector.schema()
        assert schema.columns == []


class TestLocalFileConnectorConnection:
    @pytest.mark.asyncio
    async def test_existing_file_returns_ok(self):
        connector = LocalFileConnector({"default_path": str(DATA_CSV)})
        status = await connector.test_connection()

        assert status.ok is True
        assert status.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_missing_file_returns_not_ok(self):
        connector = LocalFileConnector({"default_path": "/nonexistent/file.csv"})
        status = await connector.test_connection()
        assert status.ok is False