import pytest
import asyncio
import asyncmy
import asyncmy.cursors
import polars as pl

from dataprep.connectors.mysql import MySQLConnector

MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "database": "testdb",
    "user": "testuser",
    "password": "testpass",
    "default_schema": "testdb",
    "default_table": "test_table",
}

_POLARS_TO_MYSQL = {
    "Int8": "TINYINT",
    "Int16": "SMALLINT",
    "Int32": "INT",
    "Int64": "BIGINT",
    "UInt8": "TINYINT UNSIGNED",
    "UInt16": "SMALLINT UNSIGNED",
    "UInt32": "INT UNSIGNED",
    "UInt64": "BIGINT UNSIGNED",
    "Float32": "FLOAT",
    "Float64": "DOUBLE",
    "Boolean": "BOOLEAN",
    "Utf8": "TEXT",
    "String": "TEXT",
    "Date": "DATE",
    "Datetime": "DATETIME",
}

DROP_TABLE = "DROP TABLE IF EXISTS testdb.test_table"

_READY_RETRIES = 20
_READY_INTERVAL = 3


def _build_create_table(df: pl.DataFrame) -> str:
    col_defs = []
    for name, dtype in zip(df.columns, df.dtypes):
        mysql_type = _POLARS_TO_MYSQL.get(str(dtype), "TEXT")
        col_defs.append(f"`{name}` {mysql_type}")
    return f"CREATE TABLE IF NOT EXISTS testdb.test_table ({', '.join(col_defs)})"


async def _raw_conn() -> asyncmy.Connection:
    last_exc: Exception | None = None
    for attempt in range(_READY_RETRIES):
        try:
            conn = await asyncmy.connect(
                host=MYSQL_CONFIG["host"],
                port=MYSQL_CONFIG["port"],
                db=MYSQL_CONFIG["database"],
                user=MYSQL_CONFIG["user"],
                password=MYSQL_CONFIG["password"],
            )
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
            return conn
        except Exception as exc:
            last_exc = exc
            print(f"MySQL not ready, retry {attempt + 1}/{_READY_RETRIES}...")
            await asyncio.sleep(_READY_INTERVAL)

    raise RuntimeError(
        f"MySQL at {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']} not accepting "
        f"queries after {_READY_RETRIES * _READY_INTERVAL}s: {last_exc}"
    )


@pytest.fixture(scope="module", autouse=True)
async def mysql_ready():
    await _raw_conn()


@pytest.fixture(scope="module")
async def mysql_table(sample_df: pl.DataFrame):
    conn = await _raw_conn()
    async with conn.cursor() as cur:
        await cur.execute(DROP_TABLE)
        await cur.execute(_build_create_table(sample_df))
        cols = sample_df.columns
        placeholders = ",".join(["%s"] * len(cols))
        rows = [tuple(r) for r in sample_df.iter_rows()]
        await cur.executemany(
            f"INSERT INTO testdb.test_table ({','.join(f'`{c}`' for c in cols)}) VALUES ({placeholders})",
            rows,
        )
    await conn.commit()
    await conn.ensure_closed()
    yield
    conn = await _raw_conn()
    async with conn.cursor() as cur:
        await cur.execute(DROP_TABLE)
    await conn.commit()
    await conn.ensure_closed()


@pytest.fixture
def connector():
    return MySQLConnector(MYSQL_CONFIG)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_test_connection(connector: MySQLConnector):
    status = await connector.test_connection()
    assert status.ok
    assert status.latency_ms is not None
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_returns_all_rows(connector: MySQLConnector, mysql_table, sample_df: pl.DataFrame):
    df = await connector.read({"table": "test_table", "schema": "testdb"})
    assert len(df) == len(sample_df)
    assert set(df.columns) == set(sample_df.columns)
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_with_sql(connector: MySQLConnector, mysql_table, sample_df: pl.DataFrame):
    first_col = sample_df.columns[0]
    second_col = sample_df.columns[1]
    first_val = sample_df[first_col][0]
    expected_second = sample_df[second_col][0]

    df = await connector.read({
        "sql": f"SELECT `{first_col}`, `{second_col}` FROM testdb.test_table WHERE `{first_col}` = {first_val!r}"
    })
    assert len(df) >= 1
    assert df[second_col][0] == expected_second
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_stream_yields_batches(connector: MySQLConnector, mysql_table, sample_df: pl.DataFrame):
    batch_size = max(1, len(sample_df) // 3)
    frames = []
    async for chunk in connector.read_stream(
        {"table": "test_table", "schema": "testdb"},
        {"batch_size": batch_size},
    ):
        frames.append(chunk)

    assert len(frames) >= 2
    assert sum(len(f) for f in frames) == len(sample_df)
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_schema(connector: MySQLConnector, mysql_table, sample_df: pl.DataFrame):
    info = await connector.schema()
    names = [c.name for c in info.columns]
    assert set(names) == set(sample_df.columns)
    assert info.row_count_estimate == len(sample_df)
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_schema_dtype_mapping(connector: MySQLConnector, mysql_table, sample_df: pl.DataFrame):
    info = await connector.schema()
    by_name = {c.name: c.dtype for c in info.columns}
    for col_name, polars_dtype in zip(sample_df.columns, sample_df.dtypes):
        dtype_str = str(polars_dtype)
        if dtype_str in _POLARS_TO_MYSQL:
            assert by_name[col_name] is not None
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_append(connector: MySQLConnector, mysql_table, sample_df: pl.DataFrame):
    extra_n = min(2, len(sample_df))
    extra = sample_df.head(extra_n)
    result = await connector.write(
        extra,
        {"table": "test_table", "schema": "testdb"},
        {"write_mode": "append"},
    )
    assert result.rows_written == extra_n

    df = await connector.read({"table": "test_table", "schema": "testdb"})
    assert len(df) == len(sample_df) + extra_n
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_replace(connector: MySQLConnector, mysql_table, sample_df: pl.DataFrame):
    small_n = min(3, len(sample_df))
    small = sample_df.head(small_n)
    result = await connector.write(
        small,
        {"table": "test_table", "schema": "testdb"},
        {"write_mode": "replace"},
    )
    assert result.rows_written == small_n

    df = await connector.read({"table": "test_table", "schema": "testdb"})
    assert len(df) == small_n
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_batch_size(connector: MySQLConnector, mysql_table, sample_df: pl.DataFrame):
    batch_size = max(1, len(sample_df) // 3)
    result = await connector.write(
        sample_df,
        {"table": "test_table", "schema": "testdb"},
        {"write_mode": "replace", "batch_size": batch_size},
    )
    assert result.rows_written == len(sample_df)
    await connector.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connection_failure():
    bad = MySQLConnector({**MYSQL_CONFIG, "port": 9999})
    status = await bad.test_connection()
    assert not status.ok
    assert status.error is not None