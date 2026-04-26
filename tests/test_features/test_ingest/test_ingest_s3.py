import io
import json
from pathlib import Path

import polars as pl
import pytest

from dataprep.connectors.s3 import S3Connector
from dataprep.core.context import PipelineContext
from dataprep.core.events import EventType
from dataprep.features.ingest.pipeline import IngestConfig, IngestPipeline
from dataprep.storage.artifact import ArtifactStore

pytestmark = pytest.mark.integration

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"

S3_CONFIG = {
    "aws_access_key_id": "minioadmin",
    "aws_secret_access_key": "minioadmin",
    "region": "us-east-1",
    "endpoint_url": "http://localhost:9000",
    "bucket": "testbucket",
}

BUCKET = "testbucket"
PARQUET_KEY = "ingest_test/data.parquet"
CSV_KEY = "ingest_test/data.csv"
PARQUET_URL = f"s3://{BUCKET}/{PARQUET_KEY}"
CSV_URL = f"s3://{BUCKET}/{CSV_KEY}"
GLOB_URL = f"s3://{BUCKET}/ingest_partitioned/part_*.parquet"


def _make_context(tmp_path: Path) -> PipelineContext:
    store = ArtifactStore(base_path=str(tmp_path / "artifacts"))
    return PipelineContext(artifact_store=store)


def _make_connector(**overrides) -> S3Connector:
    return S3Connector({**S3_CONFIG, **overrides})


async def _drain(pipeline: IngestPipeline, context: PipelineContext) -> list:
    plan = pipeline.plan(context)
    events = []
    async for event in pipeline.execute(plan, context):
        events.append(event)
    return events


async def _upload_parquet(connector: S3Connector, df: pl.DataFrame, key: str) -> None:
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    async with connector._client() as client:
        await client.put_object(Bucket=BUCKET, Key=key, Body=buf.read())


async def _upload_csv(connector: S3Connector, df: pl.DataFrame, key: str) -> None:
    buf = io.BytesIO()
    df.write_csv(buf)
    buf.seek(0)
    async with connector._client() as client:
        await client.put_object(Bucket=BUCKET, Key=key, Body=buf.read())


async def _delete_key(connector: S3Connector, key: str) -> None:
    async with connector._client() as client:
        try:
            await client.delete_object(Bucket=BUCKET, Key=key)
        except Exception:
            pass


@pytest.fixture()
def source_df() -> pl.DataFrame:
    return pl.read_csv(DATA_CSV, try_parse_dates=True)


@pytest.fixture()
async def parquet_object(source_df: pl.DataFrame):
    connector = _make_connector()
    await _upload_parquet(connector, source_df, PARQUET_KEY)
    yield source_df
    await _delete_key(connector, PARQUET_KEY)


@pytest.fixture()
async def csv_object(source_df: pl.DataFrame):
    connector = _make_connector()
    await _upload_csv(connector, source_df, CSV_KEY)
    yield source_df
    await _delete_key(connector, CSV_KEY)


@pytest.fixture()
async def partitioned_objects(source_df: pl.DataFrame):
    connector = _make_connector()
    mid = len(source_df) // 2
    part1 = source_df.head(mid)
    part2 = source_df.tail(len(source_df) - mid)
    await _upload_parquet(connector, part1, "ingest_partitioned/part_0.parquet")
    await _upload_parquet(connector, part2, "ingest_partitioned/part_1.parquet")
    yield source_df
    await _delete_key(connector, "ingest_partitioned/part_0.parquet")
    await _delete_key(connector, "ingest_partitioned/part_1.parquet")


class TestIngestS3Validate:
    def test_valid_config_passes(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        result = pipeline.validate(_make_context(tmp_path))
        assert result.ok
        assert result.errors == []

    def test_empty_query_fails(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={}))
        result = pipeline.validate(_make_context(tmp_path))
        assert not result.ok
        assert any("query" in e for e in result.errors)

    def test_none_connector_fails(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=None, query={"url": PARQUET_URL}))
        result = pipeline.validate(_make_context(tmp_path))
        assert not result.ok
        assert any("connector" in e for e in result.errors)


class TestIngestS3Plan:
    def test_plan_has_required_steps(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        plan = pipeline.plan(_make_context(tmp_path))
        step_names = [s["name"] for s in plan.steps]
        assert "stream_read" in step_names
        assert "materialize_parquet" in step_names
        assert "save_schema" in step_names

    def test_plan_metadata_carries_query(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        plan = pipeline.plan(_make_context(tmp_path))
        assert plan.metadata["query"]["url"] == PARQUET_URL


class TestIngestS3ExecuteParquet:
    @pytest.mark.asyncio
    async def test_event_sequence(self, parquet_object, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        events = await _drain(pipeline, _make_context(tmp_path))
        types = [e.event_type for e in events]
        assert types[0] == EventType.STARTED
        assert types[-1] == EventType.DONE
        assert EventType.ERROR not in types

    @pytest.mark.asyncio
    async def test_rows_read_matches_source(self, parquet_object, tmp_path):
        source_df = parquet_object
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        events = await _drain(pipeline, _make_context(tmp_path))
        assert events[-1].payload["rows_read"] == len(source_df)

    @pytest.mark.asyncio
    async def test_parquet_artifact_row_count(self, parquet_object, tmp_path):
        source_df = parquet_object
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        events = await _drain(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert len(saved_df) == len(source_df)

    @pytest.mark.asyncio
    async def test_parquet_artifact_columns(self, parquet_object, tmp_path):
        source_df = parquet_object
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        events = await _drain(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert set(saved_df.columns) == set(source_df.columns)

    @pytest.mark.asyncio
    async def test_schema_artifact_row_and_column_count(self, parquet_object, tmp_path):
        source_df = parquet_object
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        events = await _drain(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["row_count"] == len(source_df)
        assert schema_data["column_count"] == len(source_df.columns)

    @pytest.mark.asyncio
    async def test_schema_artifact_field_names(self, parquet_object, tmp_path):
        source_df = parquet_object
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        events = await _drain(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert set(col["name"] for col in schema_data["schema"]) == set(source_df.columns)

    @pytest.mark.asyncio
    async def test_schema_artifact_duration_non_negative(self, parquet_object, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        events = await _drain(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["ingest_duration_s"] >= 0

    @pytest.mark.asyncio
    async def test_progress_events_have_rows_read(self, parquet_object, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        events = await _drain(pipeline, _make_context(tmp_path))
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress) >= 1
        for e in progress:
            assert e.payload["rows_read"] > 0

    @pytest.mark.asyncio
    async def test_done_event_has_artifact_ids(self, parquet_object, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        events = await _drain(pipeline, _make_context(tmp_path))
        assert "output_artifact_id" in events[-1].payload
        assert "schema_artifact_id" in events[-1].payload

    @pytest.mark.asyncio
    async def test_parquet_artifact_is_on_disk(self, parquet_object, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        events = await _drain(pipeline, ctx)
        assert ctx.artifact_store.exists(events[-1].payload["output_artifact_id"], "parquet")

    @pytest.mark.asyncio
    async def test_schema_artifact_is_on_disk(self, parquet_object, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        events = await _drain(pipeline, ctx)
        assert ctx.artifact_store.exists(events[-1].payload["schema_artifact_id"], "json")


class TestIngestS3ExecuteCSV:
    @pytest.mark.asyncio
    async def test_event_sequence(self, csv_object, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": CSV_URL}))
        events = await _drain(pipeline, _make_context(tmp_path))
        types = [e.event_type for e in events]
        assert types[0] == EventType.STARTED
        assert types[-1] == EventType.DONE
        assert EventType.ERROR not in types

    @pytest.mark.asyncio
    async def test_rows_read_matches_source(self, csv_object, tmp_path):
        source_df = csv_object
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": CSV_URL}))
        events = await _drain(pipeline, _make_context(tmp_path))
        assert events[-1].payload["rows_read"] == len(source_df)

    @pytest.mark.asyncio
    async def test_parquet_artifact_row_count(self, csv_object, tmp_path):
        source_df = csv_object
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": CSV_URL}))
        events = await _drain(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert len(saved_df) == len(source_df)

    @pytest.mark.asyncio
    async def test_schema_artifact_row_and_column_count(self, csv_object, tmp_path):
        source_df = csv_object
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": CSV_URL}))
        events = await _drain(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["row_count"] == len(source_df)
        assert schema_data["column_count"] == len(source_df.columns)


class TestIngestS3ExecuteGlob:
    @pytest.mark.asyncio
    async def test_glob_rows_read_matches_source(self, partitioned_objects, tmp_path):
        source_df = partitioned_objects
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": GLOB_URL}))
        events = await _drain(pipeline, _make_context(tmp_path))
        assert events[-1].event_type == EventType.DONE
        assert events[-1].payload["rows_read"] == len(source_df)

    @pytest.mark.asyncio
    async def test_glob_parquet_artifact_row_count(self, partitioned_objects, tmp_path):
        source_df = partitioned_objects
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": GLOB_URL}))
        events = await _drain(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert len(saved_df) == len(source_df)

    @pytest.mark.asyncio
    async def test_glob_progress_events_accumulate(self, partitioned_objects, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": GLOB_URL}))
        events = await _drain(pipeline, _make_context(tmp_path))
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress) >= 2


class TestIngestS3ErrorPath:
    @pytest.mark.asyncio
    async def test_nonexistent_key_raises(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(), query={"url": f"s3://{BUCKET}/nonexistent_xyz.parquet"})
        )
        with pytest.raises(Exception):
            await _drain(pipeline, _make_context(tmp_path))

    @pytest.mark.asyncio
    async def test_error_event_emitted_before_raise(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(), query={"url": f"s3://{BUCKET}/nonexistent_xyz.parquet"})
        )
        ctx = _make_context(tmp_path)
        events = []
        try:
            plan = pipeline.plan(ctx)
            async for event in pipeline.execute(plan, ctx):
                events.append(event)
        except Exception:
            pass
        assert any(e.event_type == EventType.ERROR for e in events)

    @pytest.mark.asyncio
    async def test_bad_endpoint_raises(self, tmp_path):
        connector = _make_connector(endpoint_url="http://localhost:19999")
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"url": PARQUET_URL}))
        with pytest.raises(Exception):
            await _drain(pipeline, _make_context(tmp_path))

    @pytest.mark.asyncio
    async def test_bad_endpoint_error_event_emitted(self, tmp_path):
        connector = _make_connector(endpoint_url="http://localhost:19999")
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"url": PARQUET_URL}))
        ctx = _make_context(tmp_path)
        events = []
        try:
            plan = pipeline.plan(ctx)
            async for event in pipeline.execute(plan, ctx):
                events.append(event)
        except Exception:
            pass
        assert any(e.event_type == EventType.ERROR for e in events)

    @pytest.mark.asyncio
    async def test_bad_credentials_raises(self, tmp_path):
        connector = _make_connector(aws_secret_access_key="wrongsecret")
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"url": PARQUET_URL}))
        with pytest.raises(Exception):
            await _drain(pipeline, _make_context(tmp_path))


class TestIngestS3Serialize:
    def test_feature_field(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        assert pipeline.serialize()["feature"] == "ingest"

    def test_query_preserved(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        assert pipeline.serialize()["query"]["url"] == PARQUET_URL

    def test_options_preserved(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}, options={"format": "parquet"})
        )
        assert pipeline.serialize()["options"]["format"] == "parquet"

    def test_is_json_serializable(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=_make_connector(), query={"url": PARQUET_URL}))
        json.dumps(pipeline.serialize())