import json
from pathlib import Path

import asyncpg
import polars as pl
import pytest

from dataprep.connectors.postgresql import PostgreSQLConnector
from dataprep.core.context import PipelineContext
from dataprep.core.events import EventType
from dataprep.features.ingest.pipeline import IngestConfig, IngestPipeline
from dataprep.storage.artifact import ArtifactStore

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"

PG_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "testdb",
    "user": "testuser",
    "password": "testpass",
}

_PG_TABLE = "ingest_test"


def _make_context(tmp_path: Path) -> PipelineContext:
    store = ArtifactStore(base_path=str(tmp_path / "artifacts"))
    return PipelineContext(artifact_store=store)


async def _drain(pipeline: IngestPipeline, context: PipelineContext) -> list:
    plan = pipeline.plan(context)
    events = []
    async for event in pipeline.execute(plan, context):
        events.append(event)
    return events


def _col_defs(df: pl.DataFrame) -> str:
    parts = []
    for col, dtype in zip(df.columns, df.dtypes):
        s = str(dtype)
        if "Int" in s:
            pg = "BIGINT"
        elif "Float" in s:
            pg = "DOUBLE PRECISION"
        elif "Boolean" in s:
            pg = "BOOLEAN"
        elif "Datetime" in s:
            pg = "TIMESTAMP"
        elif "Date" in s:
            pg = "DATE"
        else:
            pg = "TEXT"
        parts.append(f'"{col}" {pg}')
    return ", ".join(parts)


@pytest.fixture()
async def pg_setup(tmp_path: Path):
    source_df = pl.read_csv(DATA_CSV, try_parse_dates=True)

    conn = await asyncpg.connect(**{k: v for k, v in PG_CONFIG.items()})
    await conn.execute(f'DROP TABLE IF EXISTS public."{_PG_TABLE}"')
    await conn.execute(f'CREATE TABLE public."{_PG_TABLE}" ({_col_defs(source_df)})')

    cols = ", ".join(f'"{c}"' for c in source_df.columns)
    placeholders = ", ".join(f"${i+1}" for i in range(len(source_df.columns)))
    insert_sql = f'INSERT INTO public."{_PG_TABLE}" ({cols}) VALUES ({placeholders})'
    await conn.executemany(insert_sql, [tuple(r) for r in source_df.iter_rows()])
    await conn.close()

    yield source_df, tmp_path

    conn = await asyncpg.connect(**{k: v for k, v in PG_CONFIG.items()})
    await conn.execute(f'DROP TABLE IF EXISTS public."{_PG_TABLE}"')
    await conn.close()


def _make_connector() -> PostgreSQLConnector:
    return PostgreSQLConnector(PG_CONFIG)


class TestIngestPostgreSQLValidate:
    def test_valid_config_passes(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        result = pipeline.validate(_make_context(tmp_path))
        assert result.ok
        assert result.errors == []

    def test_empty_query_fails(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={}))
        result = pipeline.validate(_make_context(tmp_path))
        assert not result.ok
        assert any("query" in e for e in result.errors)

    def test_none_connector_fails(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=None, query={"table": _PG_TABLE}))
        result = pipeline.validate(_make_context(tmp_path))
        assert not result.ok
        assert any("connector" in e for e in result.errors)


class TestIngestPostgreSQLPlan:
    def test_plan_has_required_steps(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        plan = pipeline.plan(_make_context(tmp_path))
        step_names = [s["name"] for s in plan.steps]
        assert "stream_read" in step_names
        assert "materialize_parquet" in step_names
        assert "save_schema" in step_names

    def test_plan_metadata_carries_query(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        plan = pipeline.plan(_make_context(tmp_path))
        assert plan.metadata["query"]["table"] == _PG_TABLE


class TestIngestPostgreSQLExecuteTable:
    @pytest.mark.asyncio
    async def test_event_sequence(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        types = [e.event_type for e in events]
        assert types[0] == EventType.STARTED
        assert types[-1] == EventType.DONE
        assert EventType.ERROR not in types

    @pytest.mark.asyncio
    async def test_rows_read_matches_source(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        assert events[-1].payload["rows_read"] == len(source_df)

    @pytest.mark.asyncio
    async def test_parquet_artifact_row_count(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert len(saved_df) == len(source_df)

    @pytest.mark.asyncio
    async def test_parquet_artifact_columns(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert saved_df.columns == source_df.columns

    @pytest.mark.asyncio
    async def test_schema_artifact_row_and_column_count(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["row_count"] == len(source_df)
        assert schema_data["column_count"] == len(source_df.columns)

    @pytest.mark.asyncio
    async def test_schema_artifact_field_names(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert [col["name"] for col in schema_data["schema"]] == source_df.columns

    @pytest.mark.asyncio
    async def test_schema_artifact_duration_non_negative(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["ingest_duration_s"] >= 0

    @pytest.mark.asyncio
    async def test_progress_events_have_rows_read(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress) >= 1
        for e in progress:
            assert e.payload["rows_read"] > 0

    @pytest.mark.asyncio
    async def test_done_event_has_artifact_ids(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        assert "output_artifact_id" in events[-1].payload
        assert "schema_artifact_id" in events[-1].payload

    @pytest.mark.asyncio
    async def test_parquet_artifact_is_on_disk(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        assert ctx.artifact_store.exists(events[-1].payload["output_artifact_id"], "parquet")

    @pytest.mark.asyncio
    async def test_schema_artifact_is_on_disk(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        assert ctx.artifact_store.exists(events[-1].payload["schema_artifact_id"], "json")

    @pytest.mark.asyncio
    async def test_custom_batch_size(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(
                connector=connector,
                query={"table": _PG_TABLE},
                options={"batch_size": max(1, len(source_df) // 3)},
            )
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        assert events[-1].event_type == EventType.DONE
        assert events[-1].payload["rows_read"] == len(source_df)


class TestIngestPostgreSQLExecuteSQL:
    @pytest.mark.asyncio
    async def test_raw_sql_query(self, pg_setup):
        source_df, tmp_path = pg_setup
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(
                connector=connector,
                query={"sql": f'SELECT * FROM public."{_PG_TABLE}"'},
            )
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        assert events[-1].event_type == EventType.DONE
        assert events[-1].payload["rows_read"] == len(source_df)

    @pytest.mark.asyncio
    async def test_sql_with_limit(self, pg_setup):
        source_df, tmp_path = pg_setup
        limit = max(1, len(source_df) // 2)
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(
                connector=connector,
                query={"sql": f'SELECT * FROM public."{_PG_TABLE}" LIMIT {limit}'},
            )
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        assert events[-1].event_type == EventType.DONE
        assert events[-1].payload["rows_read"] == limit


class TestIngestPostgreSQLErrorPath:
    @pytest.mark.asyncio
    async def test_nonexistent_table_raises(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(connector=connector, query={"table": "nonexistent_table_xyz"})
        )
        with pytest.raises(Exception):
            await _drain(pipeline, _make_context(tmp_path))
        await connector.close()

    @pytest.mark.asyncio
    async def test_error_event_emitted_before_raise(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(connector=connector, query={"table": "nonexistent_table_xyz"})
        )
        ctx = _make_context(tmp_path)
        events = []
        try:
            plan = pipeline.plan(ctx)
            async for event in pipeline.execute(plan, ctx):
                events.append(event)
        except Exception:
            pass
        await connector.close()
        assert any(e.event_type == EventType.ERROR for e in events)

    @pytest.mark.asyncio
    async def test_bad_credentials_raises(self, tmp_path):
        connector = PostgreSQLConnector({
            "host": "localhost",
            "port": 5433,
            "database": "testdb",
            "user": "wronguser",
            "password": "wrongpass",
        })
        pipeline = IngestPipeline(
            IngestConfig(connector=connector, query={"table": _PG_TABLE})
        )
        with pytest.raises(Exception):
            await _drain(pipeline, _make_context(tmp_path))
        await connector.close()


class TestIngestPostgreSQLSerialize:
    def test_feature_field(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        assert pipeline.serialize()["feature"] == "ingest"

    def test_query_preserved(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        assert pipeline.serialize()["query"]["table"] == _PG_TABLE

    def test_options_preserved(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(connector=connector, query={"table": _PG_TABLE}, options={"batch_size": 500})
        )
        assert pipeline.serialize()["options"]["batch_size"] == 500

    def test_is_json_serializable(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _PG_TABLE}))
        json.dumps(pipeline.serialize())