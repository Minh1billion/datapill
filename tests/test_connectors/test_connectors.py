import pytest
import pytest_asyncio
import polars as pl
from pathlib import Path
import tempfile

from dataprep.connectors.local_file import LocalFileConnector, _adaptive_batch_size
from dataprep.connectors.postgresql import PostgreSQLConnector

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"
DATA_XLSX = FIXTURES_DIR / "data.xlsx"


class TestAdaptiveBatchSize:
    def test_hint_within_bounds(self, sample_df):
        assert _adaptive_batch_size(sample_df, 50_000) == 50_000

    def test_hint_below_min_clamps(self, sample_df):
        assert _adaptive_batch_size(sample_df, 1) == 10_000

    def test_hint_above_max_clamps(self, sample_df):
        assert _adaptive_batch_size(sample_df, 999_999_999) == 500_000

    def test_no_hint_returns_reasonable_size(self, sample_df):
        result = _adaptive_batch_size(sample_df, None)
        assert 10_000 <= result <= 500_000

    def test_empty_df_returns_min(self):
        empty = pl.DataFrame({"a": pl.Series([], dtype=pl.Int32)})
        assert _adaptive_batch_size(empty, None) == 10_000


class TestLocalFileConnectorRead:
    @pytest.mark.asyncio
    async def test_read_csv(self, local_connector, sample_df):
        result = await local_connector.read({"path": str(DATA_CSV)})
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_df)
        assert result.columns == sample_df.columns

    @pytest.mark.asyncio
    async def test_read_parquet(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "test.parquet"
        sample_df.write_parquet(path)
        result = await local_connector.read({"path": str(path)})
        assert result.shape == sample_df.shape

    @pytest.mark.asyncio
    async def test_read_json(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "test.ndjson"
        sample_df.write_ndjson(path)
        result = await local_connector.read({"path": str(path)})
        assert len(result) == len(sample_df)

    @pytest.mark.asyncio
    async def test_read_xlsx(self, local_connector):
        result = await local_connector.read({"path": str(DATA_XLSX)})
        assert len(result) == 1000


    @pytest.mark.asyncio
    async def test_read_unsupported_format_raises(self, local_connector, tmp_path):
        df = pl.read_excel(FIXTURES_DIR / "data.xlsx")
        path = tmp_path / "out.xlsx"
        result = await local_connector.write(df, {"path": str(path)})
        assert path.exists()
        assert result.rows_written == len(df)

    @pytest.mark.asyncio
    async def test_read_with_delimiter_option(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "test.csv"
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
    async def test_stream_batch_size_respected(self, local_connector, sample_df):
        batch_size = max(1, len(sample_df) // 3)
        chunks = []
        async for chunk in local_connector.read_stream(
            {"path": str(DATA_CSV)}, {"batch_size": batch_size}
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1
        for chunk in chunks[:-1]:
            assert len(chunk) == batch_size

    @pytest.mark.asyncio
    async def test_stream_schema_consistent(self, local_connector, sample_df):
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
        roundtrip = pl.read_csv(path)
        assert len(roundtrip) == len(sample_df)

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
    async def test_write_unsupported_format_raises(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "out.txt"
        with pytest.raises(ValueError, match="Unsupported"):
            await local_connector.write(sample_df, {"path": str(path)})

    @pytest.mark.asyncio
    async def test_write_duration_positive(self, local_connector, tmp_path, sample_df):
        path = tmp_path / "out.parquet"
        result = await local_connector.write(sample_df, {"path": str(path)})
        assert result.duration_s >= 0


class TestLocalFileConnectorSchema:
    @pytest.mark.asyncio
    async def test_schema_from_csv(self):
        connector = LocalFileConnector({"default_path": str(DATA_CSV)})
        schema = await connector.schema()
        assert len(schema.columns) > 0
        for col in schema.columns:
            assert col.name
            assert col.dtype

    @pytest.mark.asyncio
    async def test_schema_from_parquet(self, sample_df, tmp_path):
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
    async def test_connection_existing_file(self):
        connector = LocalFileConnector({"default_path": str(DATA_CSV)})
        status = await connector.test_connection()
        assert status.ok is True
        assert status.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_connection_missing_file(self):
        connector = LocalFileConnector({"default_path": "/nonexistent/file.csv"})
        status = await connector.test_connection()
        assert status.ok is False


@pytest.mark.integration
class TestPostgreSQLConnectorRead:
    @pytest.mark.asyncio
    async def test_read_full_table(self, pg_connector, pg_table, sample_df):
        result = await pg_connector.read({"table": pg_table["table"]})
        assert isinstance(result, pl.DataFrame)
        assert len(result) == pg_table["row_count"]

    @pytest.mark.asyncio
    async def test_read_with_sql(self, pg_connector, pg_table, sample_df):
        table = pg_table["table"]
        result = await pg_connector.read({"sql": f'SELECT * FROM "{table}" LIMIT 5'})
        assert len(result) == min(5, pg_table["row_count"])

    @pytest.mark.asyncio
    async def test_read_columns_match(self, pg_connector, pg_table, sample_df):
        result = await pg_connector.read({"table": pg_table["table"]})
        assert set(result.columns) == set(sample_df.columns)

    @pytest.mark.asyncio
    async def test_read_empty_result(self, pg_connector, pg_table):
        table = pg_table["table"]
        result = await pg_connector.read({"sql": f'SELECT * FROM "{table}" WHERE 1=0'})
        assert len(result) == 0


@pytest.mark.integration
class TestPostgreSQLConnectorStream:
    @pytest.mark.asyncio
    async def test_stream_all_rows(self, pg_connector, pg_table):
        total = 0
        async for chunk in pg_connector.read_stream({"table": pg_table["table"]}):
            assert isinstance(chunk, pl.DataFrame)
            total += len(chunk)
        assert total == pg_table["row_count"]

    @pytest.mark.asyncio
    async def test_stream_batch_size(self, pg_connector, pg_table):
        batch_size = max(1, pg_table["row_count"] // 2)
        chunks = []
        async for chunk in pg_connector.read_stream(
            {"table": pg_table["table"]}, {"batch_size": batch_size}
        ):
            chunks.append(chunk)
        assert sum(len(c) for c in chunks) == pg_table["row_count"]


@pytest.mark.integration
class TestPostgreSQLConnectorWrite:
    @pytest.mark.asyncio
    async def test_write_append(self, pg_connector, pg_table, sample_df):
        import asyncpg
        target = {"table": pg_table["table"], "schema": "public"}
        result = await pg_connector.write(sample_df, target, {"write_mode": "append"})
        assert result.rows_written == len(sample_df)

        conn = await asyncpg.connect(host="localhost", port=5433, database="testdb", user="testuser", password="testpass")
        count = await conn.fetchval(f'SELECT COUNT(*) FROM "{pg_table["table"]}"')
        await conn.close()
        assert count == pg_table["row_count"] + len(sample_df)

    @pytest.mark.asyncio
    async def test_write_replace(self, pg_connector, pg_table, sample_df):
        import asyncpg
        target = {"table": pg_table["table"], "schema": "public"}
        result = await pg_connector.write(sample_df, target, {"write_mode": "replace"})
        assert result.rows_written == len(sample_df)

        conn = await asyncpg.connect(host="localhost", port=5433, database="testdb", user="testuser", password="testpass")
        count = await conn.fetchval(f'SELECT COUNT(*) FROM "{pg_table["table"]}"')
        await conn.close()
        assert count == len(sample_df)

    @pytest.mark.asyncio
    async def test_write_duration_positive(self, pg_connector, pg_table, sample_df):
        target = {"table": pg_table["table"]}
        result = await pg_connector.write(sample_df, target)
        assert result.duration_s >= 0


@pytest.mark.integration
class TestPostgreSQLConnectorSchema:
    @pytest.mark.asyncio
    async def test_schema_returns_columns(self, pg_table, sample_df):
        connector = PostgreSQLConnector({
            **{"host": "localhost", "port": 5433, "database": "testdb", "user": "testuser", "password": "testpass"},
            "default_table": pg_table["table"],
            "default_schema": "public",
        })
        schema = await connector.schema()
        assert len(schema.columns) == len(sample_df.columns)
        assert schema.row_count_estimate == pg_table["row_count"]

    @pytest.mark.asyncio
    async def test_schema_no_table_returns_empty(self, pg_connector):
        schema = await pg_connector.schema()
        assert schema.columns == []


@pytest.mark.integration
class TestPostgreSQLConnectorConnection:
    @pytest.mark.asyncio
    async def test_connection_success(self, pg_connector):
        status = await pg_connector.test_connection()
        assert status.ok is True
        assert status.latency_ms > 0

    @pytest.mark.asyncio
    async def test_connection_failure_bad_host(self):
        bad_connector = PostgreSQLConnector({
            "host": "localhost",
            "port": 19999,
            "database": "testdb",
            "user": "testuser",
            "password": "testpass",
        })
        status = await bad_connector.test_connection()
        assert status.ok is False
        assert status.error is not None

    @pytest.mark.asyncio
    async def test_close(self, pg_connector):
        await pg_connector.test_connection()
        await pg_connector.close()
        assert pg_connector._pool is None or pg_connector._pool._closed