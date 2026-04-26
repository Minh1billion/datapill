import asyncio
import json
from pathlib import Path

import asyncmy
import asyncmy.cursors
import polars as pl
import pytest

from dataprep.connectors.mysql import MySQLConnector
from dataprep.core.context import PipelineContext
from dataprep.core.events import EventType
from dataprep.features.ingest.pipeline import IngestConfig, IngestPipeline
from dataprep.storage.artifact import ArtifactStore

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"

MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "database": "testdb",
    "user": "testuser",
    "password": "testpass",
}

_TABLE = "ingest_test"
_SCHEMA = "testdb"

_MYSQL_READY_RETRIES = 20
_MYSQL_READY_INTERVAL = 3

_POLARS_TO_MYSQL: dict[str, str] = {
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


async def _wait_mysql_accepting() -> None:
    last_exc: Exception | None = None
    for _ in range(_MYSQL_READY_RETRIES):
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
            await conn.ensure_closed()
            return
        except Exception as exc:
            last_exc = exc
            await asyncio.sleep(_MYSQL_READY_INTERVAL)
    raise RuntimeError(
        f"MySQL at {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']} not accepting "
        f"queries after {_MYSQL_READY_RETRIES * _MYSQL_READY_INTERVAL}s: {last_exc}"
    )


@pytest.fixture(scope="module", autouse=True)
async def mysql_ready():
    await _wait_mysql_accepting()


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
        mysql_type = _POLARS_TO_MYSQL.get(str(dtype), "TEXT")
        parts.append(f"`{col}` {mysql_type}")
    return ", ".join(parts)


async def _raw_conn() -> asyncmy.Connection:
    return await asyncmy.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        db=MYSQL_CONFIG["database"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
    )


async def _create_table(source_df: pl.DataFrame) -> None:
    conn = await _raw_conn()
    async with conn.cursor() as cur:
        await cur.execute(f"DROP TABLE IF EXISTS `{_SCHEMA}`.`{_TABLE}`")
        await cur.execute(f"CREATE TABLE `{_SCHEMA}`.`{_TABLE}` ({_col_defs(source_df)})")
        cols = ", ".join(f"`{c}`" for c in source_df.columns)
        placeholders = ", ".join(["%s"] * len(source_df.columns))
        await cur.executemany(
            f"INSERT INTO `{_SCHEMA}`.`{_TABLE}` ({cols}) VALUES ({placeholders})",
            [tuple(r) for r in source_df.iter_rows()],
        )
    await conn.commit()
    await conn.ensure_closed()


async def _drop_table() -> None:
    conn = await _raw_conn()
    async with conn.cursor() as cur:
        await cur.execute(f"DROP TABLE IF EXISTS `{_SCHEMA}`.`{_TABLE}`")
    await conn.commit()
    await conn.ensure_closed()


@pytest.fixture()
def mysql_setup(tmp_path: Path):
    source_df = pl.read_csv(DATA_CSV, try_parse_dates=True)
    asyncio.get_event_loop().run_until_complete(_create_table(source_df))
    yield source_df, tmp_path
    asyncio.get_event_loop().run_until_complete(_drop_table())


def _make_connector() -> MySQLConnector:
    return MySQLConnector(MYSQL_CONFIG)


class TestIngestMySQLValidate:
    def test_valid_config_passes(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
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
        pipeline = IngestPipeline(IngestConfig(connector=None, query={"table": _TABLE}))
        result = pipeline.validate(_make_context(tmp_path))
        assert not result.ok
        assert any("connector" in e for e in result.errors)


class TestIngestMySQLPlan:
    def test_plan_has_required_steps(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        plan = pipeline.plan(_make_context(tmp_path))
        step_names = [s["name"] for s in plan.steps]
        assert "stream_read" in step_names
        assert "materialize_parquet" in step_names
        assert "save_schema" in step_names

    def test_plan_metadata_carries_query(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        plan = pipeline.plan(_make_context(tmp_path))
        assert plan.metadata["query"]["table"] == _TABLE


class TestIngestMySQLExecuteTable:
    @pytest.mark.asyncio
    async def test_event_sequence(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        types = [e.event_type for e in events]
        assert types[0] == EventType.STARTED
        assert types[-1] == EventType.DONE
        assert EventType.ERROR not in types

    @pytest.mark.asyncio
    async def test_rows_read_matches_source(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        assert events[-1].payload["rows_read"] == len(source_df)

    @pytest.mark.asyncio
    async def test_parquet_artifact_row_count(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert len(saved_df) == len(source_df)

    @pytest.mark.asyncio
    async def test_parquet_artifact_columns(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert saved_df.columns == source_df.columns

    @pytest.mark.asyncio
    async def test_schema_artifact_row_and_column_count(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["row_count"] == len(source_df)
        assert schema_data["column_count"] == len(source_df.columns)

    @pytest.mark.asyncio
    async def test_schema_artifact_field_names(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert [col["name"] for col in schema_data["schema"]] == source_df.columns

    @pytest.mark.asyncio
    async def test_schema_artifact_duration_non_negative(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["ingest_duration_s"] >= 0

    @pytest.mark.asyncio
    async def test_progress_events_have_rows_read(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress) >= 1
        for e in progress:
            assert e.payload["rows_read"] > 0

    @pytest.mark.asyncio
    async def test_done_event_has_artifact_ids(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        assert "output_artifact_id" in events[-1].payload
        assert "schema_artifact_id" in events[-1].payload

    @pytest.mark.asyncio
    async def test_parquet_artifact_is_on_disk(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        assert ctx.artifact_store.exists(events[-1].payload["output_artifact_id"], "parquet")

    @pytest.mark.asyncio
    async def test_schema_artifact_is_on_disk(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        events = await _drain(pipeline, ctx)
        await connector.close()
        assert ctx.artifact_store.exists(events[-1].payload["schema_artifact_id"], "json")

    @pytest.mark.asyncio
    async def test_custom_batch_size(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(
                connector=connector,
                query={"table": _TABLE, "schema": _SCHEMA},
                options={"batch_size": max(1, len(source_df) // 3)},
            )
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        assert events[-1].event_type == EventType.DONE
        assert events[-1].payload["rows_read"] == len(source_df)


class TestIngestMySQLExecuteSQL:
    @pytest.mark.asyncio
    async def test_raw_sql_query(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(
                connector=connector,
                query={"sql": f"SELECT * FROM `{_SCHEMA}`.`{_TABLE}`"},
            )
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        assert events[-1].event_type == EventType.DONE
        assert events[-1].payload["rows_read"] == len(source_df)

    @pytest.mark.asyncio
    async def test_sql_with_limit(self, mysql_setup):
        source_df, tmp_path = mysql_setup
        limit = max(1, len(source_df) // 2)
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(
                connector=connector,
                query={"sql": f"SELECT * FROM `{_SCHEMA}`.`{_TABLE}` LIMIT {limit}"},
            )
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        await connector.close()
        assert events[-1].event_type == EventType.DONE
        assert events[-1].payload["rows_read"] == limit


class TestIngestMySQLErrorPath:
    @pytest.mark.asyncio
    async def test_nonexistent_table_raises(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(connector=connector, query={"table": "nonexistent_xyz", "schema": _SCHEMA})
        )
        with pytest.raises(Exception):
            await _drain(pipeline, _make_context(tmp_path))
        await connector.close()

    @pytest.mark.asyncio
    async def test_error_event_emitted_before_raise(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(connector=connector, query={"table": "nonexistent_xyz", "schema": _SCHEMA})
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
        connector = MySQLConnector({**MYSQL_CONFIG, "password": "wrongpass"})
        pipeline = IngestPipeline(
            IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA})
        )
        with pytest.raises(Exception):
            await _drain(pipeline, _make_context(tmp_path))
        await connector.close()


class TestIngestMySQLSerialize:
    def test_feature_field(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        assert pipeline.serialize()["feature"] == "ingest"

    def test_query_preserved(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        assert pipeline.serialize()["query"]["table"] == _TABLE

    def test_options_preserved(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(
            IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}, options={"batch_size": 500})
        )
        assert pipeline.serialize()["options"]["batch_size"] == 500

    def test_is_json_serializable(self, tmp_path):
        connector = _make_connector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"table": _TABLE, "schema": _SCHEMA}))
        json.dumps(pipeline.serialize())