import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator

import asyncmy
import asyncmy.cursors
import polars as pl
import pytest
import pytest_asyncio

from dataprep.connectors.mysql import MySQLConnector

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"

MYSQL_CONFIG: dict = {
    "host": "localhost",
    "port": 3307,
    "database": "testdb",
    "user": "testuser",
    "password": "testpass",
}


def _asyncmy_kwargs(config: dict) -> dict:
    return {
        "host": config["host"],
        "port": config["port"],
        "db": config["database"],
        "user": config["user"],
        "password": config["password"],
    }


def _wait_for_mysql(timeout: float = 60) -> None:
    last_exc = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        loop = asyncio.new_event_loop()
        try:
            async def _check():
                conn = await asyncmy.connect(**_asyncmy_kwargs(MYSQL_CONFIG))
                async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
                    await cur.execute("SELECT 1")
                conn.close()
            loop.run_until_complete(_check())
            return
        except Exception as e:
            last_exc = e
            time.sleep(1)
        finally:
            loop.close()
    raise RuntimeError(f"MySQL did not become ready in time: {last_exc}")


@pytest.fixture(scope="session", autouse=True)
def wait_for_mysql():
    _wait_for_mysql()


@pytest.fixture(scope="session")
def sample_df() -> pl.DataFrame:
    return pl.read_csv(DATA_CSV, try_parse_dates=True)


@pytest.fixture
def mysql_connector() -> MySQLConnector:
    return MySQLConnector(MYSQL_CONFIG)


@pytest_asyncio.fixture(scope="function")
async def mysql_table(sample_df: pl.DataFrame) -> AsyncGenerator[dict, None]:
    table_name = f"test_data_{int(time.time() * 1000)}"
    conn = await asyncmy.connect(**_asyncmy_kwargs(MYSQL_CONFIG))
    async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
        ddl = _df_to_mysql_ddl(sample_df)
        await cur.execute(f"CREATE TABLE `{table_name}` ({ddl})")
        await _insert_rows(cur, table_name, sample_df)
    await conn.commit()
    conn.close()

    yield {"table": table_name, "schema": "testdb", "row_count": len(sample_df)}

    conn = await asyncmy.connect(**_asyncmy_kwargs(MYSQL_CONFIG))
    async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
        await cur.execute(f"DROP TABLE IF EXISTS `{table_name}`")
    await conn.commit()
    conn.close()


async def _insert_rows(cur, table: str, df: pl.DataFrame) -> None:
    cols = df.columns
    col_names = ", ".join(f"`{c}`" for c in cols)
    placeholders = ", ".join(["%s"] * len(cols))
    sql = f"INSERT INTO `{table}` ({col_names}) VALUES ({placeholders})"
    await cur.executemany(sql, [tuple(row) for row in df.iter_rows()])


def _df_to_mysql_ddl(df: pl.DataFrame) -> str:
    _TYPE_MAP = {
        "Int8": "TINYINT",
        "Int16": "SMALLINT",
        "Int32": "INT",
        "Int64": "BIGINT",
        "UInt8": "SMALLINT",
        "UInt16": "INT",
        "UInt32": "BIGINT",
        "UInt64": "BIGINT",
        "Float32": "FLOAT",
        "Float64": "DOUBLE",
        "Boolean": "BOOLEAN",
        "Utf8": "TEXT",
        "String": "TEXT",
        "Date": "DATE",
        "Datetime": "DATETIME",
    }
    parts = [
        f"`{col}` {_TYPE_MAP.get(str(dtype), 'TEXT')}"
        for col, dtype in zip(df.columns, df.dtypes)
    ]
    return ", ".join(parts)


@pytest.mark.integration
class TestMySQLConnectorRead:
    @pytest.mark.asyncio
    async def test_read_full_table_returns_all_rows(self, mysql_connector, mysql_table):
        result = await mysql_connector.read({"table": mysql_table["table"]})
        assert isinstance(result, pl.DataFrame)
        assert len(result) == mysql_table["row_count"]

    @pytest.mark.asyncio
    async def test_read_with_sql_respects_limit(self, mysql_connector, mysql_table):
        table = mysql_table["table"]
        result = await mysql_connector.read({"sql": f"SELECT * FROM `{table}` LIMIT 5"})
        assert len(result) == min(5, mysql_table["row_count"])

    @pytest.mark.asyncio
    async def test_read_columns_match_source(self, mysql_connector, mysql_table, sample_df):
        result = await mysql_connector.read({"table": mysql_table["table"]})
        assert set(result.columns) == set(sample_df.columns)

    @pytest.mark.asyncio
    async def test_read_empty_result_from_false_filter(self, mysql_connector, mysql_table):
        table = mysql_table["table"]
        result = await mysql_connector.read({"sql": f"SELECT * FROM `{table}` WHERE 1=0"})
        assert len(result) == 0


@pytest.mark.integration
class TestMySQLConnectorStream:
    @pytest.mark.asyncio
    async def test_stream_yields_all_rows(self, mysql_connector, mysql_table):
        total = 0
        async for chunk in mysql_connector.read_stream({"table": mysql_table["table"]}):
            assert isinstance(chunk, pl.DataFrame)
            total += len(chunk)
        assert total == mysql_table["row_count"]

    @pytest.mark.asyncio
    async def test_stream_total_rows_with_custom_batch_size(self, mysql_connector, mysql_table):
        batch_size = max(1, mysql_table["row_count"] // 2)
        chunks = []
        async for chunk in mysql_connector.read_stream(
            {"table": mysql_table["table"]}, {"batch_size": batch_size}
        ):
            chunks.append(chunk)
        assert sum(len(c) for c in chunks) == mysql_table["row_count"]

    @pytest.mark.asyncio
    async def test_stream_schema_is_consistent_across_chunks(self, mysql_connector, mysql_table):
        schemas = []
        async for chunk in mysql_connector.read_stream({"table": mysql_table["table"]}):
            schemas.append(chunk.schema)
        assert all(s == schemas[0] for s in schemas)


@pytest.mark.integration
class TestMySQLConnectorWrite:
    @pytest.mark.asyncio
    async def test_append_increases_row_count(self, mysql_connector, mysql_table, sample_df):
        target = {"table": mysql_table["table"], "schema": mysql_table["schema"]}
        result = await mysql_connector.write(sample_df, target, {"write_mode": "append"})
        assert result.rows_written == len(sample_df)

        count = await _fetch_count(mysql_table["table"])
        assert count == mysql_table["row_count"] + len(sample_df)

    @pytest.mark.asyncio
    async def test_replace_resets_row_count(self, mysql_connector, mysql_table, sample_df):
        target = {"table": mysql_table["table"], "schema": mysql_table["schema"]}
        result = await mysql_connector.write(sample_df, target, {"write_mode": "replace"})
        assert result.rows_written == len(sample_df)

        count = await _fetch_count(mysql_table["table"])
        assert count == len(sample_df)

    @pytest.mark.asyncio
    async def test_write_result_duration_is_non_negative(self, mysql_connector, mysql_table, sample_df):
        result = await mysql_connector.write(sample_df, {"table": mysql_table["table"]})
        assert result.duration_s >= 0


@pytest.mark.integration
class TestMySQLConnectorSchema:
    @pytest.mark.asyncio
    async def test_schema_column_count_matches_source(self, mysql_table, sample_df):
        connector = MySQLConnector(
            {**MYSQL_CONFIG, "default_table": mysql_table["table"], "default_schema": "testdb"}
        )
        schema = await connector.schema()
        assert len(schema.columns) == len(sample_df.columns)
        assert schema.row_count_estimate == mysql_table["row_count"]

    @pytest.mark.asyncio
    async def test_schema_without_table_returns_empty(self, mysql_connector):
        schema = await mysql_connector.schema()
        assert schema.columns == []


@pytest.mark.integration
class TestMySQLConnectorConnection:
    @pytest.mark.asyncio
    async def test_valid_config_returns_ok(self, mysql_connector):
        status = await mysql_connector.test_connection()
        assert status.ok is True
        assert status.latency_ms > 0

    @pytest.mark.asyncio
    async def test_bad_port_returns_not_ok_with_error(self):
        connector = MySQLConnector({**MYSQL_CONFIG, "port": 19999})
        status = await connector.test_connection()
        assert status.ok is False
        assert status.error is not None

    @pytest.mark.asyncio
    async def test_close_releases_pool(self, mysql_connector):
        await mysql_connector.test_connection()
        await mysql_connector.close()
        assert mysql_connector._pool is None or mysql_connector._pool._closed


async def _fetch_count(table: str) -> int:
    conn = await asyncmy.connect(**_asyncmy_kwargs(MYSQL_CONFIG))
    async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
        await cur.execute(f"SELECT COUNT(*) AS n FROM `{table}`")
        row = await cur.fetchone()
    conn.close()
    return row["n"]