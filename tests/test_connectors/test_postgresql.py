import pytest
import polars as pl
import asyncpg

from dataprep.connectors.postgresql import PostgreSQLConnector

PG_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "testdb",
    "user": "testuser",
    "password": "testpass",
    "default_schema": "public",
    "default_table": "test_table",
}

_POLARS_TO_PG = {
    "Int8": "SMALLINT",
    "Int16": "SMALLINT",
    "Int32": "INTEGER",
    "Int64": "BIGINT",
    "UInt8": "SMALLINT",
    "UInt16": "INTEGER",
    "UInt32": "BIGINT",
    "UInt64": "BIGINT",
    "Float32": "REAL",
    "Float64": "DOUBLE PRECISION",
    "Boolean": "BOOLEAN",
    "Utf8": "TEXT",
    "String": "TEXT",
    "Date": "DATE",
    "Datetime": "TIMESTAMP",
}

DROP_TABLE = "DROP TABLE IF EXISTS public.test_table"


def _build_create_table(df: pl.DataFrame) -> str:
    col_defs = []
    for name, dtype in zip(df.columns, df.dtypes):
        pg_type = _POLARS_TO_PG.get(str(dtype), "TEXT")
        col_defs.append(f'"{name}" {pg_type}')
    return f"CREATE TABLE IF NOT EXISTS public.test_table ({', '.join(col_defs)})"


def _build_insert(df: pl.DataFrame) -> str:
    cols = ", ".join(f'"{c}"' for c in df.columns)
    placeholders = ", ".join(f"${i+1}" for i in range(len(df.columns)))
    return f"INSERT INTO public.test_table ({cols}) VALUES ({placeholders})"


@pytest.fixture(scope="module")
async def pg_table(sample_df: pl.DataFrame):
    conn = await asyncpg.connect(
        host=PG_CONFIG["host"],
        port=PG_CONFIG["port"],
        database=PG_CONFIG["database"],
        user=PG_CONFIG["user"],
        password=PG_CONFIG["password"],
    )
    await conn.execute(DROP_TABLE)
    await conn.execute(_build_create_table(sample_df))
    rows = [tuple(r) for r in sample_df.iter_rows()]
    await conn.executemany(_build_insert(sample_df), rows)
    await conn.close()
    yield
    conn = await asyncpg.connect(
        host=PG_CONFIG["host"],
        port=PG_CONFIG["port"],
        database=PG_CONFIG["database"],
        user=PG_CONFIG["user"],
        password=PG_CONFIG["password"],
    )
    await conn.execute(DROP_TABLE)
    await conn.close()


@pytest.fixture
def connector():
    return PostgreSQLConnector(PG_CONFIG)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_test_connection(connector: PostgreSQLConnector):
    status = await connector.test_connection()
    assert status.ok
    assert status.latency_ms is not None
    assert status.latency_ms >= 0
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_returns_all_rows(connector: PostgreSQLConnector, pg_table, sample_df: pl.DataFrame):
    df = await connector.read({"table": "test_table", "schema": "public"})
    assert len(df) == len(sample_df)
    assert set(df.columns) == set(sample_df.columns)
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_with_sql(connector: PostgreSQLConnector, pg_table, sample_df: pl.DataFrame):
    first_col = sample_df.columns[0]
    second_col = sample_df.columns[1]
    first_val = sample_df[first_col][0]
    expected_second = sample_df[second_col][0]

    df = await connector.read({
        "sql": f'SELECT "{first_col}", "{second_col}" FROM public.test_table WHERE "{first_col}" = {first_val!r}'
    })
    assert len(df) >= 1
    assert df[first_col][0] == first_val
    assert df[second_col][0] == expected_second
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_stream_yields_all_rows(connector: PostgreSQLConnector, pg_table, sample_df: pl.DataFrame):
    batch_size = max(1, len(sample_df) // 3)
    frames = []
    async for chunk in connector.read_stream(
        {"table": "test_table", "schema": "public"},
        {"batch_size": batch_size},
    ):
        frames.append(chunk)

    assert len(frames) > 1
    assert sum(len(f) for f in frames) == len(sample_df)
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_schema(connector: PostgreSQLConnector, pg_table, sample_df: pl.DataFrame):
    info = await connector.schema()
    names = [c.name for c in info.columns]
    assert set(names) == set(sample_df.columns)
    assert info.row_count_estimate == len(sample_df)
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_append(connector: PostgreSQLConnector, pg_table, sample_df: pl.DataFrame):
    extra_n = min(2, len(sample_df))
    extra = sample_df.head(extra_n)
    result = await connector.write(
        extra,
        {"table": "test_table", "schema": "public"},
        {"write_mode": "append"},
    )
    assert result.rows_written == extra_n

    df = await connector.read({"table": "test_table", "schema": "public"})
    assert len(df) == len(sample_df) + extra_n
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_replace(connector: PostgreSQLConnector, pg_table, sample_df: pl.DataFrame):
    small_n = min(3, len(sample_df))
    small = sample_df.head(small_n)
    result = await connector.write(
        small,
        {"table": "test_table", "schema": "public"},
        {"write_mode": "replace"},
    )
    assert result.rows_written == small_n

    df = await connector.read({"table": "test_table", "schema": "public"})
    assert len(df) == small_n
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_returns_duration(connector: PostgreSQLConnector, pg_table, sample_df: pl.DataFrame):
    result = await connector.write(
        sample_df.head(1),
        {"table": "test_table", "schema": "public"},
    )
    assert result.duration_s >= 0
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connection_failure():
    bad = PostgreSQLConnector({**PG_CONFIG, "port": 9999})
    status = await bad.test_connection()
    assert not status.ok
    assert status.error is not None