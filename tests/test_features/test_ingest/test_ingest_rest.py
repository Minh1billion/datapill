import json
from pathlib import Path

import polars as pl
import pytest
from pytest_httpserver import HTTPServer

from dataprep.connectors.rest import RESTConnector
from dataprep.core.context import PipelineContext
from dataprep.core.events import EventType
from dataprep.features.ingest.pipeline import IngestConfig, IngestPipeline
from dataprep.storage.artifact import ArtifactStore


def _make_rows(n: int = 10) -> list[dict]:
    return [{"id": i, "name": f"item_{i}", "value": i * 1.5} for i in range(1, n + 1)]


SAMPLE_ROWS = _make_rows(10)
PAGE_SIZE = len(SAMPLE_ROWS) // 2


def _make_config(server: HTTPServer, **extra) -> dict:
    return {
        "base_url": server.url_for(""),
        "headers": {"Content-Type": "application/json"},
        "timeout_seconds": 5,
        "max_retries": 2,
        **extra,
    }


def _make_context(tmp_path: Path) -> PipelineContext:
    store = ArtifactStore(base_path=str(tmp_path / "artifacts"))
    return PipelineContext(artifact_store=store)


def _make_connector(server: HTTPServer, **extra) -> RESTConnector:
    return RESTConnector(_make_config(server, **extra))


async def _drain(pipeline: IngestPipeline, context: PipelineContext) -> list:
    plan = pipeline.plan(context)
    events = []
    async for event in pipeline.execute(plan, context):
        events.append(event)
    return events


@pytest.fixture
def flat_server(httpserver: HTTPServer):
    httpserver.expect_request("/items").respond_with_json(SAMPLE_ROWS)
    return httpserver


@pytest.fixture
def nested_server(httpserver: HTTPServer):
    httpserver.expect_request("/data").respond_with_json({"results": SAMPLE_ROWS})
    return httpserver


@pytest.fixture
def paginated_server(httpserver: HTTPServer):
    page1 = SAMPLE_ROWS[:PAGE_SIZE]
    page2 = SAMPLE_ROWS[PAGE_SIZE:]
    httpserver.expect_request("/items", query_string=f"limit={PAGE_SIZE}&offset=0").respond_with_json(page1)
    httpserver.expect_request("/items", query_string=f"limit={PAGE_SIZE}&offset={PAGE_SIZE}").respond_with_json(page2)
    httpserver.expect_request("/items", query_string=f"limit={PAGE_SIZE}&offset={len(SAMPLE_ROWS)}").respond_with_json([])
    return httpserver


@pytest.fixture
def retry_server(httpserver: HTTPServer):
    call_count = {"n": 0}

    def handler(request):
        from werkzeug.wrappers import Response
        call_count["n"] += 1
        if call_count["n"] < 2:
            return Response(status=429, headers={"Retry-After": "0"})
        return Response(
            response=json.dumps(SAMPLE_ROWS),
            status=200,
            content_type="application/json",
        )

    httpserver.expect_request("/items").respond_with_handler(handler)
    return httpserver, call_count


class TestIngestRESTValidate:
    def test_valid_config_passes(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        result = pipeline.validate(_make_context(tmp_path))
        assert result.ok
        assert result.errors == []

    def test_empty_query_fails(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={})
        )
        result = pipeline.validate(_make_context(tmp_path))
        assert not result.ok
        assert any("query" in e for e in result.errors)

    def test_none_connector_fails(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=None, query={"endpoint": "/items"}))
        result = pipeline.validate(_make_context(tmp_path))
        assert not result.ok
        assert any("connector" in e for e in result.errors)


class TestIngestRESTplan:
    def test_plan_has_required_steps(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        plan = pipeline.plan(_make_context(tmp_path))
        step_names = [s["name"] for s in plan.steps]
        assert "stream_read" in step_names
        assert "materialize_parquet" in step_names
        assert "save_schema" in step_names

    def test_plan_metadata_carries_query(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        plan = pipeline.plan(_make_context(tmp_path))
        assert plan.metadata["query"]["endpoint"] == "/items"


class TestIngestRESTExecuteFlat:
    @pytest.mark.asyncio
    async def test_event_sequence(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        types = [e.event_type for e in events]
        assert types[0] == EventType.STARTED
        assert types[-1] == EventType.DONE
        assert EventType.ERROR not in types

    @pytest.mark.asyncio
    async def test_rows_read_matches_source(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        assert events[-1].payload["rows_read"] == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_parquet_artifact_row_count(self, flat_server, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        events = await _drain(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert len(saved_df) == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_parquet_artifact_columns(self, flat_server, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        events = await _drain(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert set(saved_df.columns) == set(SAMPLE_ROWS[0].keys())

    @pytest.mark.asyncio
    async def test_schema_artifact_row_and_column_count(self, flat_server, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        events = await _drain(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["row_count"] == len(SAMPLE_ROWS)
        assert schema_data["column_count"] == len(SAMPLE_ROWS[0])

    @pytest.mark.asyncio
    async def test_schema_artifact_field_names(self, flat_server, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        events = await _drain(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert set(col["name"] for col in schema_data["schema"]) == set(SAMPLE_ROWS[0].keys())

    @pytest.mark.asyncio
    async def test_schema_artifact_duration_non_negative(self, flat_server, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        events = await _drain(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["ingest_duration_s"] >= 0

    @pytest.mark.asyncio
    async def test_progress_events_have_rows_read(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress) >= 1
        for e in progress:
            assert e.payload["rows_read"] > 0

    @pytest.mark.asyncio
    async def test_done_event_has_artifact_ids(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        assert "output_artifact_id" in events[-1].payload
        assert "schema_artifact_id" in events[-1].payload

    @pytest.mark.asyncio
    async def test_parquet_artifact_is_on_disk(self, flat_server, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        events = await _drain(pipeline, ctx)
        assert ctx.artifact_store.exists(events[-1].payload["output_artifact_id"], "parquet")

    @pytest.mark.asyncio
    async def test_schema_artifact_is_on_disk(self, flat_server, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        events = await _drain(pipeline, ctx)
        assert ctx.artifact_store.exists(events[-1].payload["schema_artifact_id"], "json")


class TestIngestRESTExecuteNested:
    @pytest.mark.asyncio
    async def test_rows_read_matches_source(self, nested_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(nested_server, response_path="results"),
                query={"endpoint": "/data"},
            )
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        assert events[-1].payload["rows_read"] == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_parquet_artifact_row_count(self, nested_server, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(nested_server, response_path="results"),
                query={"endpoint": "/data"},
            )
        )
        events = await _drain(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert len(saved_df) == len(SAMPLE_ROWS)


class TestIngestRESTExecutePaginated:
    @pytest.mark.asyncio
    async def test_rows_read_matches_source(self, paginated_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(
                    paginated_server,
                    pagination={"type": "offset", "limit": PAGE_SIZE, "limit_param": "limit", "offset_param": "offset"},
                ),
                query={"endpoint": "/items"},
            )
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        assert events[-1].payload["rows_read"] == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_progress_events_per_page(self, paginated_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(
                    paginated_server,
                    pagination={"type": "offset", "limit": PAGE_SIZE, "limit_param": "limit", "offset_param": "offset"},
                ),
                query={"endpoint": "/items"},
            )
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress) >= 2

    @pytest.mark.asyncio
    async def test_parquet_artifact_row_count(self, paginated_server, tmp_path):
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(
                    paginated_server,
                    pagination={"type": "offset", "limit": PAGE_SIZE, "limit_param": "limit", "offset_param": "offset"},
                ),
                query={"endpoint": "/items"},
            )
        )
        events = await _drain(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert len(saved_df) == len(SAMPLE_ROWS)


class TestIngestRESTExecuteRetry:
    @pytest.mark.asyncio
    async def test_succeeds_after_429(self, retry_server, tmp_path):
        httpserver, call_count = retry_server
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(httpserver, max_retries=3),
                query={"endpoint": "/items"},
            )
        )
        events = await _drain(pipeline, _make_context(tmp_path))
        assert events[-1].event_type == EventType.DONE
        assert events[-1].payload["rows_read"] == len(SAMPLE_ROWS)
        assert call_count["n"] >= 2


class TestIngestRESTErrorPath:
    @pytest.mark.asyncio
    async def test_bad_endpoint_raises(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/nonexistent"})
        )
        with pytest.raises(Exception):
            await _drain(pipeline, _make_context(tmp_path))

    @pytest.mark.asyncio
    async def test_error_event_emitted_before_raise(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/nonexistent"})
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
    async def test_unreachable_host_raises(self, tmp_path):
        connector = RESTConnector({
            "base_url": "http://localhost:19999",
            "timeout_seconds": 1,
            "max_retries": 1,
        })
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"endpoint": "/items"}))
        with pytest.raises(Exception):
            await _drain(pipeline, _make_context(tmp_path))

    @pytest.mark.asyncio
    async def test_unreachable_host_error_event_emitted(self, tmp_path):
        connector = RESTConnector({
            "base_url": "http://localhost:19999",
            "timeout_seconds": 1,
            "max_retries": 1,
        })
        pipeline = IngestPipeline(IngestConfig(connector=connector, query={"endpoint": "/items"}))
        ctx = _make_context(tmp_path)
        events = []
        try:
            plan = pipeline.plan(ctx)
            async for event in pipeline.execute(plan, ctx):
                events.append(event)
        except Exception:
            pass
        assert any(e.event_type == EventType.ERROR for e in events)


class TestIngestRESTSerialize:
    def test_feature_field(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        assert pipeline.serialize()["feature"] == "ingest"

    def test_query_preserved(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        assert pipeline.serialize()["query"]["endpoint"] == "/items"

    def test_options_preserved(self, flat_server, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(flat_server),
                query={"endpoint": "/items"},
                options={"batch_size": 100},
            )
        )
        assert pipeline.serialize()["options"]["batch_size"] == 100

    def test_is_json_serializable(self, flat_server, tmp_path):
        import json
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(flat_server), query={"endpoint": "/items"})
        )
        json.dumps(pipeline.serialize())