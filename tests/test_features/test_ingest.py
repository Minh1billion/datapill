import pytest
import polars as pl
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from dataprep.features.ingest.pipeline import IngestPipeline, IngestConfig, _extract_schema
from dataprep.connectors.local_file import LocalFileConnector
from dataprep.connectors.postgresql import PostgreSQLConnector
from dataprep.core.events import EventType
from dataprep.core.interfaces import ExecutionPlan

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"


def _collect_events(async_gen):
    import asyncio
    async def _inner():
        events = []
        async for e in async_gen:
            events.append(e)
        return events
    return asyncio.get_event_loop().run_until_complete(_inner())


class TestExtractSchema:
    def test_returns_all_columns(self, sample_df):
        schema = _extract_schema(sample_df)
        assert len(schema) == len(sample_df.columns)

    def test_schema_fields_present(self, sample_df):
        schema = _extract_schema(sample_df)
        for entry in schema:
            assert "name" in entry
            assert "dtype" in entry
            assert "nullable" in entry

    def test_nullable_flag(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
        schema = _extract_schema(df)
        nullable_map = {e["name"]: e["nullable"] for e in schema}
        assert nullable_map["a"] is True
        assert nullable_map["b"] is False


class TestIngestPipelineValidation:
    def test_valid_config_passes(self, local_connector, pipeline_context):
        config = IngestConfig(connector=local_connector, query={"path": str(DATA_CSV)})
        pipeline = IngestPipeline(config)
        result = pipeline.validate(pipeline_context)
        assert result.ok is True
        assert result.errors == []

    def test_empty_query_fails(self, local_connector, pipeline_context):
        config = IngestConfig(connector=local_connector, query={})
        pipeline = IngestPipeline(config)
        result = pipeline.validate(pipeline_context)
        assert result.ok is False
        assert any("query" in e for e in result.errors)

    def test_missing_connector_fails(self, pipeline_context):
        config = IngestConfig(connector=None, query={"path": str(DATA_CSV)})
        pipeline = IngestPipeline(config)
        result = pipeline.validate(pipeline_context)
        assert result.ok is False
        assert any("connector" in e for e in result.errors)


class TestIngestPipelinePlan:
    def test_plan_has_required_steps(self, local_connector, pipeline_context):
        config = IngestConfig(connector=local_connector, query={"path": str(DATA_CSV)})
        pipeline = IngestPipeline(config)
        plan = pipeline.plan(pipeline_context)
        step_names = [s["name"] for s in plan.steps]
        assert "stream_read" in step_names
        assert "materialize_parquet" in step_names
        assert "save_schema" in step_names

    def test_plan_metadata_contains_query(self, local_connector, pipeline_context):
        query = {"path": str(DATA_CSV)}
        config = IngestConfig(connector=local_connector, query=query)
        pipeline = IngestPipeline(config)
        plan = pipeline.plan(pipeline_context)
        assert plan.metadata["query"] == query


class TestIngestPipelineExecute:
    @pytest.mark.asyncio
    async def test_execute_emits_started(self, local_connector, pipeline_context):
        config = IngestConfig(connector=local_connector, query={"path": str(DATA_CSV)})
        pipeline = IngestPipeline(config)
        plan = pipeline.plan(pipeline_context)
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        assert events[0].event_type == EventType.STARTED

    @pytest.mark.asyncio
    async def test_execute_emits_done(self, local_connector, pipeline_context):
        config = IngestConfig(connector=local_connector, query={"path": str(DATA_CSV)})
        pipeline = IngestPipeline(config)
        plan = pipeline.plan(pipeline_context)
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        assert events[-1].event_type == EventType.DONE

    @pytest.mark.asyncio
    async def test_execute_saves_parquet_artifact(self, local_connector, pipeline_context, sample_df):
        config = IngestConfig(connector=local_connector, query={"path": str(DATA_CSV)})
        pipeline = IngestPipeline(config)
        plan = pipeline.plan(pipeline_context)
        async for _ in pipeline.execute(plan, pipeline_context):
            pass
        output_id = "test_run_ingest_output"
        assert pipeline_context.artifact_store.exists(output_id, "parquet")
        df = pipeline_context.artifact_store.load_parquet(output_id)
        assert len(df) == len(sample_df)

    @pytest.mark.asyncio
    async def test_execute_saves_schema_artifact(self, local_connector, pipeline_context, sample_df):
        config = IngestConfig(connector=local_connector, query={"path": str(DATA_CSV)})
        pipeline = IngestPipeline(config)
        plan = pipeline.plan(pipeline_context)
        async for _ in pipeline.execute(plan, pipeline_context):
            pass
        schema_id = "test_run_ingest_schema"
        assert pipeline_context.artifact_store.exists(schema_id, "json")
        meta = await pipeline_context.artifact_store.load_json(schema_id)
        assert meta["row_count"] == len(sample_df)
        assert meta["column_count"] == len(sample_df.columns)
        assert isinstance(meta["schema"], list)

    @pytest.mark.asyncio
    async def test_execute_done_payload_has_artifact_ids(self, local_connector, pipeline_context):
        config = IngestConfig(connector=local_connector, query={"path": str(DATA_CSV)})
        pipeline = IngestPipeline(config)
        plan = pipeline.plan(pipeline_context)
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        done_event = events[-1]
        assert "output_artifact_id" in done_event.payload
        assert "schema_artifact_id" in done_event.payload
        assert "rows_read" in done_event.payload

    @pytest.mark.asyncio
    async def test_execute_progress_pct_increases(self, local_connector, pipeline_context):
        config = IngestConfig(connector=local_connector, query={"path": str(DATA_CSV)})
        pipeline = IngestPipeline(config)
        plan = pipeline.plan(pipeline_context)
        pcts = []
        async for e in pipeline.execute(plan, pipeline_context):
            pcts.append(e.progress_pct)
        valid = [p for p in pcts if p is not None]
        assert valid[-1] == 1.0

    @pytest.mark.asyncio
    async def test_execute_connector_error_emits_error_event(self, pipeline_context):
        bad_connector = LocalFileConnector()
        config = IngestConfig(connector=bad_connector, query={"path": "/nonexistent/file.csv"})
        pipeline = IngestPipeline(config)
        plan = pipeline.plan(pipeline_context)
        events = []
        with pytest.raises(Exception):
            async for e in pipeline.execute(plan, pipeline_context):
                events.append(e)
        assert any(e.event_type == EventType.ERROR for e in events)

    @pytest.mark.asyncio
    async def test_execute_with_options(self, pipeline_context, sample_df, tmp_path):
        path = tmp_path / "delim.csv"
        sample_df.write_csv(path, separator=";")
        connector = LocalFileConnector()
        config = IngestConfig(
            connector=connector,
            query={"path": str(path)},
            options={"delimiter": ";"},
        )
        pipeline = IngestPipeline(config)
        plan = pipeline.plan(pipeline_context)
        async for _ in pipeline.execute(plan, pipeline_context):
            pass
        df = pipeline_context.artifact_store.load_parquet("test_run_ingest_output")
        assert len(df) == len(sample_df)


class TestIngestPipelineSerialize:
    def test_serialize_has_feature_key(self, local_connector):
        config = IngestConfig(connector=local_connector, query={"path": str(DATA_CSV)})
        pipeline = IngestPipeline(config)
        data = pipeline.serialize()
        assert data["feature"] == "ingest"
        assert "version" in data
        assert "query" in data


@pytest.mark.integration
class TestIngestPipelineWithPostgres:
    @pytest.mark.asyncio
    async def test_ingest_from_postgres(self, pg_connector, pg_table, pipeline_context, sample_df):
        config = IngestConfig(
            connector=pg_connector,
            query={"table": pg_table["table"]},
        )
        pipeline = IngestPipeline(config)
        plan = pipeline.plan(pipeline_context)
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        assert events[-1].event_type == EventType.DONE
        df = pipeline_context.artifact_store.load_parquet("test_run_ingest_output")
        assert len(df) == pg_table["row_count"]
        assert set(df.columns) == set(sample_df.columns)