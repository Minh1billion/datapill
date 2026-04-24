import asyncio
import subprocess
import time
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Any
import pytest
import pytest_asyncio
import polars as pl
import asyncpg

from dataprep.connectors.base import BaseConnector, SchemaInfo, ColumnMeta, ConnectionStatus, WriteResult
from dataprep.connectors.local_file import LocalFileConnector
from dataprep.connectors.postgresql import PostgreSQLConnector
from dataprep.storage.artifact import ArtifactStore
from dataprep.core.context import PipelineContext

PG_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "testdb",
    "user": "testuser",
    "password": "testpass",
}

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"
DATA_XLSX = FIXTURES_DIR / "data.xlsx"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as requiring Docker/PostgreSQL")


def pytest_sessionstart(session):
    result = subprocess.run(
        ["docker", "compose", "-f", "docker-compose.test.yml", "up", "-d", "--wait"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start Docker services:\n{result.stderr}")

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            conn = asyncio.get_event_loop().run_until_complete(
                asyncpg.connect(**PG_CONFIG)
            )
            asyncio.get_event_loop().run_until_complete(conn.close())
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("PostgreSQL did not become ready in time")


def pytest_sessionfinish(session, exitstatus):
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.test.yml", "down", "-v"],
        capture_output=True,
    )


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def sample_df() -> pl.DataFrame:
    return pl.read_csv(DATA_CSV, try_parse_dates=True)


@pytest_asyncio.fixture(scope="function")
async def pg_table(sample_df: pl.DataFrame) -> AsyncGenerator[dict, None]:
    table_name = f"test_data_{int(time.time() * 1000)}"
    conn = await asyncpg.connect(**PG_CONFIG)

    col_defs = _polars_schema_to_pg_ddl(sample_df)
    await conn.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

    cols = sample_df.columns
    placeholders = ", ".join(f"${i+1}" for i in range(len(cols)))
    col_names = ", ".join(f'"{c}"' for c in cols)
    insert_sql = f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})'
    records = [tuple(row) for row in sample_df.iter_rows()]
    await conn.executemany(insert_sql, records)
    await conn.close()

    yield {"table": table_name, "schema": "public", "row_count": len(sample_df)}

    conn = await asyncpg.connect(**PG_CONFIG)
    await conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    await conn.close()


@pytest.fixture
def pg_connector() -> PostgreSQLConnector:
    return PostgreSQLConnector(PG_CONFIG)


@pytest.fixture
def local_connector() -> LocalFileConnector:
    return LocalFileConnector()


@pytest.fixture
def tmp_artifact_store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(base_path=str(tmp_path / "artifacts"))


@pytest.fixture
def pipeline_context(tmp_artifact_store: ArtifactStore) -> PipelineContext:
    return PipelineContext(
        run_id="test_run",
        artifact_store=tmp_artifact_store,
    )


def _polars_schema_to_pg_ddl(df: pl.DataFrame) -> str:
    type_map = {
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
    parts = []
    for col, dtype in zip(df.columns, df.dtypes):
        pg_type = type_map.get(str(dtype), "TEXT")
        parts.append(f'"{col}" {pg_type}')
    return ", ".join(parts)