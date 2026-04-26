import json
import uuid
from pathlib import Path

import polars as pl
import pytest

from dataprep.connectors.local_file import LocalFileConnector
from dataprep.core.context import PipelineContext
from dataprep.core.events import EventType
from dataprep.features.ingest.pipeline import IngestConfig, IngestPipeline
from dataprep.storage.artifact import ArtifactStore


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"


def _make_context(tmp_path: Path) -> PipelineContext:
    store = ArtifactStore(base_path=str(tmp_path / "artifacts"))
    return PipelineContext(artifact_store=store)


async def _drain(pipeline: IngestPipeline, context: PipelineContext) -> list:
    plan = pipeline.plan(context)
    events = []
    async for event in pipeline.execute(plan, context):
        events.append(event)
    return events


@pytest.fixture(scope="module")
def source_df() -> pl.DataFrame:
    return pl.read_csv(DATA_CSV, try_parse_dates=True)


@pytest.fixture()
def tmp_fixtures(tmp_path: Path, source_df: pl.DataFrame) -> dict:
    csv_path = tmp_path / "data.csv"
    source_df.write_csv(csv_path)

    parquet_path = tmp_path / "data.parquet"
    source_df.write_parquet(parquet_path)

    ndjson_path = tmp_path / "data.ndjson"
    source_df.write_ndjson(ndjson_path)

    return {
        "csv": csv_path,
        "parquet": parquet_path,
        "ndjson": ndjson_path,
    }


class TestIngestLocalFileValidate:
    def test_valid_config_passes(self, tmp_fixtures, tmp_path):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        result = pipeline.validate(_make_context(tmp_path))
        assert result.ok
        assert result.errors == []

    def test_empty_query_fails(self, tmp_path):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={}))
        result = pipeline.validate(_make_context(tmp_path))
        assert not result.ok
        assert any("query" in e for e in result.errors)

    def test_none_connector_fails(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=None, query={"path": "x.csv"}))
        result = pipeline.validate(_make_context(tmp_path))
        assert not result.ok
        assert any("connector" in e for e in result.errors)


class TestIngestLocalFilePlan:
    def test_plan_has_required_steps(self, tmp_fixtures, tmp_path):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        plan = pipeline.plan(_make_context(tmp_path))
        step_names = [s["name"] for s in plan.steps]
        assert "stream_read" in step_names
        assert "materialize_parquet" in step_names
        assert "save_schema" in step_names

    def test_plan_metadata_carries_query(self, tmp_fixtures, tmp_path):
        path = str(tmp_fixtures["csv"])
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": path}))
        plan = pipeline.plan(_make_context(tmp_path))
        assert plan.metadata["query"]["path"] == path


class TestIngestLocalFileCSV:
    @pytest.mark.asyncio
    async def test_event_sequence(self, tmp_fixtures, tmp_path, source_df):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        events = await _drain(pipeline, _make_context(tmp_path))
        types = [e.event_type for e in events]
        assert types[0] == EventType.STARTED
        assert types[-1] == EventType.DONE
        assert EventType.ERROR not in types

    @pytest.mark.asyncio
    async def test_rows_read_matches_source(self, tmp_fixtures, tmp_path, source_df):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        events = await _drain(pipeline, _make_context(tmp_path))
        assert events[-1].payload["rows_read"] == len(source_df)

    @pytest.mark.asyncio
    async def test_parquet_artifact_shape(self, tmp_fixtures, tmp_path, source_df):
        connector = LocalFileConnector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        events = await _drain(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert saved_df.shape == source_df.shape

    @pytest.mark.asyncio
    async def test_schema_artifact_row_and_column_count(self, tmp_fixtures, tmp_path, source_df):
        connector = LocalFileConnector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        events = await _drain(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["row_count"] == len(source_df)
        assert schema_data["column_count"] == len(source_df.columns)

    @pytest.mark.asyncio
    async def test_schema_artifact_field_names(self, tmp_fixtures, tmp_path, source_df):
        connector = LocalFileConnector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        events = await _drain(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        saved_names = [col["name"] for col in schema_data["schema"]]
        assert saved_names == source_df.columns

    @pytest.mark.asyncio
    async def test_schema_artifact_duration_non_negative(self, tmp_fixtures, tmp_path):
        connector = LocalFileConnector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        events = await _drain(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["ingest_duration_s"] >= 0

    @pytest.mark.asyncio
    async def test_progress_events_have_rows_read(self, tmp_fixtures, tmp_path):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        events = await _drain(pipeline, _make_context(tmp_path))
        progress_events = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress_events) >= 1
        for e in progress_events:
            assert e.payload["rows_read"] > 0

    @pytest.mark.asyncio
    async def test_done_event_has_artifact_ids(self, tmp_fixtures, tmp_path):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        events = await _drain(pipeline, _make_context(tmp_path))
        done = events[-1]
        assert "output_artifact_id" in done.payload
        assert "schema_artifact_id" in done.payload

    @pytest.mark.asyncio
    async def test_custom_batch_size(self, tmp_fixtures, tmp_path, source_df):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(
            IngestConfig(
                connector=connector,
                query={"path": str(tmp_fixtures["csv"])},
                options={"batch_size": max(1, len(source_df) // 3)},
            )
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        assert events[-1].event_type == EventType.DONE
        assert events[-1].payload["rows_read"] == len(source_df)

    @pytest.mark.asyncio
    async def test_parquet_artifact_is_on_disk(self, tmp_fixtures, tmp_path):
        connector = LocalFileConnector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        events = await _drain(pipeline, ctx)
        assert ctx.artifact_store.exists(events[-1].payload["output_artifact_id"], "parquet")

    @pytest.mark.asyncio
    async def test_schema_artifact_is_on_disk(self, tmp_fixtures, tmp_path):
        connector = LocalFileConnector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        events = await _drain(pipeline, ctx)
        assert ctx.artifact_store.exists(events[-1].payload["schema_artifact_id"], "json")


class TestIngestLocalFileParquet:
    @pytest.mark.asyncio
    async def test_rows_read_matches_source(self, tmp_fixtures, tmp_path, source_df):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["parquet"])}))
        events = await _drain(pipeline, _make_context(tmp_path))
        assert events[-1].event_type == EventType.DONE
        assert events[-1].payload["rows_read"] == len(source_df)

    @pytest.mark.asyncio
    async def test_artifact_shape_matches(self, tmp_fixtures, tmp_path, source_df):
        connector = LocalFileConnector()
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["parquet"])}))
        events = await _drain(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert saved_df.shape == source_df.shape


class TestIngestLocalFileNDJSON:
    @pytest.mark.asyncio
    async def test_rows_read_matches_source(self, tmp_fixtures, tmp_path, source_df):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["ndjson"])}))
        events = await _drain(pipeline, _make_context(tmp_path))
        assert events[-1].event_type == EventType.DONE
        assert events[-1].payload["rows_read"] == len(source_df)


class TestIngestLocalFileErrorPath:
    @pytest.mark.asyncio
    async def test_missing_file_raises(self, tmp_path):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": "/nonexistent/data.csv"}))
        with pytest.raises(Exception):
            await _drain(pipeline, _make_context(tmp_path))

    @pytest.mark.asyncio
    async def test_error_event_emitted_before_raise(self, tmp_path):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": "/nonexistent/data.csv"}))
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
    async def test_error_event_is_terminal(self, tmp_path):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": "/nonexistent/data.csv"}))
        ctx = _make_context(tmp_path)
        events = []
        try:
            plan = pipeline.plan(ctx)
            async for event in pipeline.execute(plan, ctx):
                events.append(event)
        except Exception:
            pass
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        assert all(e.is_terminal() for e in error_events)


class TestIngestLocalFileSerialize:
    def test_feature_field(self, tmp_fixtures):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        assert pipeline.serialize()["feature"] == "ingest"

    def test_query_preserved(self, tmp_fixtures):
        path = str(tmp_fixtures["csv"])
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": path}))
        assert pipeline.serialize()["query"]["path"] == path

    def test_options_preserved(self, tmp_fixtures):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(
            IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}, options={"batch_size": 500})
        )
        assert pipeline.serialize()["options"]["batch_size"] == 500

    def test_is_json_serializable(self, tmp_fixtures):
        connector = LocalFileConnector()
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"path": str(tmp_fixtures["csv"])}))
        json.dumps(pipeline.serialize())