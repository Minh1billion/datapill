import asyncpg
import polars as pl
import pytest

from dataprep.connectors.postgresql import PostgreSQLConnector

PG_CONFIG: dict = {
    "host": "localhost",
    "port": 5433,
    "database": "testdb",
    "user": "testuser",
    "password": "testpass",
}


@pytest.mark.integration
class TestPostgreSQLConnectorRead:
    @pytest.mark.asyncio
    async def test_read_full_table_returns_all_rows(self, pg_connector, pg_table):
        result = await pg_connector.read({"table": pg_table["table"]})

        assert isinstance(result, pl.DataFrame)
        assert len(result) == pg_table["row_count"]

    @pytest.mark.asyncio
    async def test_read_with_sql_respects_limit(self, pg_connector, pg_table):
        table = pg_table["table"]
        result = await pg_connector.read({"sql": f'SELECT * FROM "{table}" LIMIT 5'})
        assert len(result) == min(5, pg_table["row_count"])

    @pytest.mark.asyncio
    async def test_read_columns_match_source_dataframe(self, pg_connector, pg_table, sample_df):
        result = await pg_connector.read({"table": pg_table["table"]})
        assert set(result.columns) == set(sample_df.columns)

    @pytest.mark.asyncio
    async def test_read_empty_result_from_false_filter(self, pg_connector, pg_table):
        table = pg_table["table"]
        result = await pg_connector.read({"sql": f'SELECT * FROM "{table}" WHERE 1=0'})
        assert len(result) == 0


@pytest.mark.integration
class TestPostgreSQLConnectorStream:
    @pytest.mark.asyncio
    async def test_stream_yields_all_rows(self, pg_connector, pg_table):
        total = 0
        async for chunk in pg_connector.read_stream({"table": pg_table["table"]}):
            assert isinstance(chunk, pl.DataFrame)
            total += len(chunk)
        assert total == pg_table["row_count"]

    @pytest.mark.asyncio
    async def test_stream_total_rows_with_custom_batch_size(self, pg_connector, pg_table):
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
    async def test_append_increases_row_count(self, pg_connector, pg_table, sample_df):
        target = {"table": pg_table["table"], "schema": "public"}
        result = await pg_connector.write(sample_df, target, {"write_mode": "append"})
        assert result.rows_written == len(sample_df)

        count = await _fetch_count(pg_table["table"])
        assert count == pg_table["row_count"] + len(sample_df)

    @pytest.mark.asyncio
    async def test_replace_resets_row_count(self, pg_connector, pg_table, sample_df):
        target = {"table": pg_table["table"], "schema": "public"}
        result = await pg_connector.write(sample_df, target, {"write_mode": "replace"})
        assert result.rows_written == len(sample_df)

        count = await _fetch_count(pg_table["table"])
        assert count == len(sample_df)

    @pytest.mark.asyncio
    async def test_write_result_duration_is_non_negative(self, pg_connector, pg_table, sample_df):
        result = await pg_connector.write(sample_df, {"table": pg_table["table"]})
        assert result.duration_s >= 0


@pytest.mark.integration
class TestPostgreSQLConnectorSchema:
    @pytest.mark.asyncio
    async def test_schema_column_count_matches_source(self, pg_table, sample_df):
        connector = PostgreSQLConnector(
            {**PG_CONFIG, "default_table": pg_table["table"], "default_schema": "public"}
        )
        schema = await connector.schema()

        assert len(schema.columns) == len(sample_df.columns)
        assert schema.row_count_estimate == pg_table["row_count"]

    @pytest.mark.asyncio
    async def test_schema_without_table_returns_empty(self, pg_connector):
        schema = await pg_connector.schema()
        assert schema.columns == []


@pytest.mark.integration
class TestPostgreSQLConnectorConnection:
    @pytest.mark.asyncio
    async def test_valid_config_returns_ok(self, pg_connector):
        status = await pg_connector.test_connection()
        assert status.ok is True
        assert status.latency_ms > 0

    @pytest.mark.asyncio
    async def test_bad_port_returns_not_ok_with_error(self):
        connector = PostgreSQLConnector({**PG_CONFIG, "port": 19999})
        status = await connector.test_connection()

        assert status.ok is False
        assert status.error is not None

    @pytest.mark.asyncio
    async def test_close_releases_pool(self, pg_connector):
        await pg_connector.test_connection()
        await pg_connector.close()
        assert pg_connector._pool is None or pg_connector._pool._closed


async def _fetch_count(table: str) -> int:
    conn = await asyncpg.connect(**PG_CONFIG)
    try:
        return await conn.fetchval(f'SELECT COUNT(*) FROM "{table}"')
    finally:
        await conn.close()