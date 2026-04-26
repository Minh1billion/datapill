from pathlib import Path

import polars as pl
import pytest

from dataprep.features.export.pipeline import ExportPipeline, WriteConfig


@pytest.fixture
def df():
    return pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "carol"],
        "score": [10.0, 20.0, 30.0],
    })


@pytest.fixture
def pg_connector():
    return {
        "source": "postgresql",
        "host": "localhost",
        "port": 5433,
        "database": "testdb",
        "user": "testuser",
        "password": "testpass",
        "table": "export_test",
    }


def test_export_csv(df, tmp_path):
    path = tmp_path / "out.csv"
    cfg = WriteConfig(format="csv", path=path)
    result = ExportPipeline(cfg).run(df)
    assert result.rows_written == 3
    assert path.exists()


def test_export_parquet(df, tmp_path):
    path = tmp_path / "out.parquet"
    cfg = WriteConfig(format="parquet", path=path, options={"compression": "snappy"})
    result = ExportPipeline(cfg).run(df)
    assert result.rows_written == 3
    loaded = pl.read_parquet(path)
    assert len(loaded) == 3


def test_dry_run_no_file(df, tmp_path):
    path = tmp_path / "out.csv"
    cfg = WriteConfig(format="csv", path=path)
    result = ExportPipeline(cfg).run(df, dry_run=True)
    assert result.rows_written == 0
    assert not path.exists()


def test_export_no_path_no_connector_raises(df):
    cfg = WriteConfig(format="csv")
    with pytest.raises(ValueError, match="path or connector_config"):
        ExportPipeline(cfg).run(df)


def test_invalid_write_mode_raises(df):
    cfg = WriteConfig(
        format="csv",
        connector_config={"source": "unsupported_source"},
        write_mode="invalid",
    )
    with pytest.raises(ValueError):
        ExportPipeline(cfg).run(df)


@pytest.mark.integration
class TestPostgresWriteback:
    @pytest.fixture(autouse=True)
    def require_postgres(self):
        import socket
        try:
            with socket.create_connection(("localhost", 5433), timeout=2):
                pass
        except OSError:
            pytest.skip("PostgreSQL not reachable")

    @pytest.fixture(autouse=True)
    def create_table(self, pg_connector):
        import asyncio
        import asyncpg

        async def _setup():
            conn = await asyncpg.connect(
                host=pg_connector["host"], port=pg_connector["port"],
                database=pg_connector["database"],
                user=pg_connector["user"], password=pg_connector["password"],
            )
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS export_test (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    score DOUBLE PRECISION
                )
            """)
            await conn.execute("TRUNCATE TABLE export_test")
            await conn.close()

        asyncio.run(_setup())

    def test_pg_replace(self, df, pg_connector):
        cfg = WriteConfig(format="parquet", connector_config=pg_connector, write_mode="replace")
        result = ExportPipeline(cfg).run(df)
        assert result.rows_written == 3

    def test_pg_append(self, df, pg_connector):
        cfg = WriteConfig(format="parquet", connector_config=pg_connector, write_mode="append")

        ExportPipeline(cfg).run(df)

        df2 = pl.DataFrame({
            "id": [4, 5, 6],
            "name": ["dave", "eve", "frank"],
            "score": [40.0, 50.0, 60.0],
        })
        ExportPipeline(cfg).run(df2)

        import asyncio
        import asyncpg

        async def _count():
            conn = await asyncpg.connect(
                host=pg_connector["host"], port=pg_connector["port"],
                database=pg_connector["database"],
                user=pg_connector["user"], password=pg_connector["password"],
            )
            count = await conn.fetchval("SELECT COUNT(*) FROM export_test")
            await conn.close()
            return count

        assert asyncio.run(_count()) == 6

    def test_pg_upsert(self, df, pg_connector):
        cfg_insert = WriteConfig(
            format="parquet", connector_config=pg_connector,
            write_mode="replace",
        )
        ExportPipeline(cfg_insert).run(df)

        updated = pl.DataFrame({"id": [1, 2], "name": ["ALICE", "BOB"], "score": [99.0, 88.0]})
        cfg_upsert = WriteConfig(
            format="parquet", connector_config=pg_connector,
            write_mode="upsert", primary_keys=["id"],
        )
        ExportPipeline(cfg_upsert).run(updated)

        import asyncio
        import asyncpg

        async def _fetch():
            conn = await asyncpg.connect(
                host=pg_connector["host"], port=pg_connector["port"],
                database=pg_connector["database"],
                user=pg_connector["user"], password=pg_connector["password"],
            )
            rows = await conn.fetch("SELECT id, name, score FROM export_test ORDER BY id")
            await conn.close()
            return rows

        rows = asyncio.run(_fetch())
        assert len(rows) == 3
        assert rows[0]["name"] == "ALICE"
        assert rows[1]["name"] == "BOB"
        assert rows[2]["name"] == "carol"

    def test_upsert_requires_primary_keys(self, df, pg_connector):
        cfg = WriteConfig(
            format="parquet", connector_config=pg_connector,
            write_mode="upsert", primary_keys=[],
        )
        with pytest.raises(ValueError, match="primary_keys"):
            ExportPipeline(cfg).run(df)