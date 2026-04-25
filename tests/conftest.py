import asyncio
import subprocess
import time
from pathlib import Path
from typing import AsyncGenerator

import asyncpg
import polars as pl
import pytest
import pytest_asyncio

from dataprep.connectors.local_file import LocalFileConnector
from dataprep.connectors.postgresql import PostgreSQLConnector
from dataprep.core.context import PipelineContext
from dataprep.storage.artifact import ArtifactStore


FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"
DATA_XLSX = FIXTURES_DIR / "data.xlsx"


PG_CONFIG: dict = {
    "host": "localhost",
    "port": 5433,
    "database": "testdb",
    "user": "testuser",
    "password": "testpass",
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: mark test as requiring Docker/PostgreSQL"
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    result = subprocess.run(
        ["docker", "compose", "-f", "docker-compose.test.yml", "up", "-d", "--wait"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start Docker services:\n{result.stderr}")

    _wait_for_postgres(timeout=30)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.test.yml", "down", "-v"],
        capture_output=True,
    )


def _wait_for_postgres(timeout: float = 30) -> None:
    loop = asyncio.get_event_loop()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = loop.run_until_complete(asyncpg.connect(**PG_CONFIG))
            loop.run_until_complete(conn.close())
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("PostgreSQL did not become ready in time")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def sample_df() -> pl.DataFrame:
    return pl.read_csv(DATA_CSV, try_parse_dates=True)


@pytest.fixture
def local_connector() -> LocalFileConnector:
    return LocalFileConnector()


@pytest.fixture
def pg_connector() -> PostgreSQLConnector:
    return PostgreSQLConnector(PG_CONFIG)


@pytest_asyncio.fixture(scope="function")
async def pg_table(sample_df: pl.DataFrame) -> AsyncGenerator[dict, None]:
    table_name = f"test_data_{int(time.time() * 1000)}"
    conn = await asyncpg.connect(**PG_CONFIG)

    await conn.execute(f'CREATE TABLE "{table_name}" ({_df_to_pg_ddl(sample_df)})')
    await _insert_rows(conn, table_name, sample_df)
    await conn.close()

    yield {"table": table_name, "schema": "public", "row_count": len(sample_df)}

    conn = await asyncpg.connect(**PG_CONFIG)
    await conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    await conn.close()


async def _insert_rows(conn: asyncpg.Connection, table: str, df: pl.DataFrame) -> None:
    cols = df.columns
    placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
    col_names = ", ".join(f'"{c}"' for c in cols)
    sql = f'INSERT INTO "{table}" ({col_names}) VALUES ({placeholders})'
    await conn.executemany(sql, [tuple(row) for row in df.iter_rows()])


def _df_to_pg_ddl(df: pl.DataFrame) -> str:
    _TYPE_MAP = {
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
    parts = [
        f'"{col}" {_TYPE_MAP.get(str(dtype), "TEXT")}'
        for col, dtype in zip(df.columns, df.dtypes)
    ]
    return ", ".join(parts)


@pytest.fixture
def tmp_artifact_store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(base_path=str(tmp_path / "artifacts"))


@pytest.fixture
def pipeline_context(tmp_artifact_store: ArtifactStore) -> PipelineContext:
    return PipelineContext(run_id="test_run", artifact_store=tmp_artifact_store)