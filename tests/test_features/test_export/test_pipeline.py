import asyncio

import polars as pl
import pytest

from dataprep.features.export.pipeline import ExportPipeline, WriteConfig


def run(pipeline, df, **kwargs):
    return asyncio.run(pipeline.run(df, **kwargs))


@pytest.fixture(scope="class")
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


def test_export_csv(sample_df, tmp_path):
    path = tmp_path / "out.csv"
    cfg = WriteConfig(format="csv", path=path)
    result = run(ExportPipeline(cfg), sample_df)
    assert result.rows_written == len(sample_df)
    assert path.exists()


def test_export_parquet(sample_df, tmp_path):
    path = tmp_path / "out.parquet"
    cfg = WriteConfig(format="parquet", path=path, options={"compression": "snappy"})
    result = run(ExportPipeline(cfg), sample_df)
    assert result.rows_written == len(sample_df)
    loaded = pl.read_parquet(path)
    assert len(loaded) == len(sample_df)


def test_dry_run_no_file(sample_df, tmp_path):
    path = tmp_path / "out.csv"
    cfg = WriteConfig(format="csv", path=path)
    result = run(ExportPipeline(cfg), sample_df, dry_run=True)
    assert result.rows_written == 0
    assert not path.exists()


def test_export_no_path_no_connector_raises(sample_df):
    cfg = WriteConfig(format="csv")
    with pytest.raises(ValueError, match="path or connector_config"):
        run(ExportPipeline(cfg), sample_df)


def test_invalid_write_mode_raises(sample_df):
    cfg = WriteConfig(
        format="csv",
        connector_config={"source": "unsupported_source"},
        write_mode="invalid",
    )
    with pytest.raises(ValueError):
        run(ExportPipeline(cfg), sample_df)


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

    @pytest.fixture(scope="class", autouse=True)
    def create_table(self, pg_connector):
        async def _setup():
            import asyncpg
            conn = await asyncpg.connect(
                host=pg_connector["host"], port=pg_connector["port"],
                database=pg_connector["database"],
                user=pg_connector["user"], password=pg_connector["password"],
            )
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS export_test (
                    id BIGINT PRIMARY KEY,
                    name TEXT,
                    category TEXT,
                    value DOUBLE PRECISION,
                    quantity BIGINT,
                    is_active BOOLEAN,
                    score DOUBLE PRECISION,
                    created_date TEXT,
                    region TEXT,
                    tag TEXT
                )
            """)
            await conn.close()

        asyncio.run(_setup())

    @pytest.fixture(autouse=True)
    def truncate_table(self, pg_connector):
        async def _truncate():
            import asyncpg
            conn = await asyncpg.connect(
                host=pg_connector["host"], port=pg_connector["port"],
                database=pg_connector["database"],
                user=pg_connector["user"], password=pg_connector["password"],
            )
            await conn.execute("TRUNCATE TABLE export_test")
            await conn.close()

        asyncio.run(_truncate())

    @pytest.fixture
    def pg_df(self, sample_df):
        return sample_df.with_columns(
            pl.col("created_date").cast(pl.Utf8).alias("created_date")
        )

    def test_pg_replace(self, pg_df, pg_connector):
        cfg = WriteConfig(format="parquet", connector_config=pg_connector, write_mode="replace")
        result = run(ExportPipeline(cfg), pg_df)
        assert result.rows_written == len(pg_df)

    def test_pg_append(self, pg_df, pg_connector):
        cfg = WriteConfig(format="parquet", connector_config=pg_connector, write_mode="append")
        run(ExportPipeline(cfg), pg_df)

        max_id = pg_df["id"].max()
        df2 = pg_df.head(10).with_columns(
            (pl.col("id") + max_id).alias("id")
        )
        run(ExportPipeline(cfg), df2)

        async def _count():
            import asyncpg
            conn = await asyncpg.connect(
                host=pg_connector["host"], port=pg_connector["port"],
                database=pg_connector["database"],
                user=pg_connector["user"], password=pg_connector["password"],
            )
            count = await conn.fetchval("SELECT COUNT(*) FROM export_test")
            await conn.close()
            return count

        assert asyncio.run(_count()) == len(pg_df) + 10

    def test_pg_upsert(self, pg_df, pg_connector):
        cfg_insert = WriteConfig(
            format="parquet", connector_config=pg_connector,
            write_mode="replace",
        )
        run(ExportPipeline(cfg_insert), pg_df)

        updated = pg_df.head(2).with_columns(pl.lit("UPDATED").alias("name"))
        cfg_upsert = WriteConfig(
            format="parquet", connector_config=pg_connector,
            write_mode="upsert", primary_keys=["id"],
        )
        run(ExportPipeline(cfg_upsert), updated)

        async def _fetch():
            import asyncpg
            conn = await asyncpg.connect(
                host=pg_connector["host"], port=pg_connector["port"],
                database=pg_connector["database"],
                user=pg_connector["user"], password=pg_connector["password"],
            )
            rows = await conn.fetch(
                "SELECT id, name FROM export_test ORDER BY id LIMIT 2"
            )
            await conn.close()
            return rows

        rows = asyncio.run(_fetch())
        assert rows[0]["name"] == "UPDATED"
        assert rows[1]["name"] == "UPDATED"

    def test_upsert_requires_primary_keys(self, pg_df, pg_connector):
        cfg = WriteConfig(
            format="parquet", connector_config=pg_connector,
            write_mode="upsert", primary_keys=[],
        )
        with pytest.raises(ValueError, match="primary_keys"):
            run(ExportPipeline(cfg), pg_df)