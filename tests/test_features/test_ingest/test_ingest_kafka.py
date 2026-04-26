import asyncio
import json
from pathlib import Path

import polars as pl
import pytest
from aiokafka import AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError

from dataprep.connectors.kafka import KafkaConnector
from dataprep.core.context import PipelineContext
from dataprep.core.events import EventType
from dataprep.features.ingest.pipeline import IngestConfig, IngestPipeline
from dataprep.storage.artifact import ArtifactStore

pytestmark = pytest.mark.integration

BOOTSTRAP = "localhost:9092"

KAFKA_CONFIG = {
    "bootstrap_servers": BOOTSTRAP,
    "group_id": "ingest-test-group",
    "auto_offset_reset": "earliest",
    "value_format": "json",
}

_TOPIC = "ingest_test_topic"
_ALL_TOPICS = [_TOPIC]

_KAFKA_READY_RETRIES = 20
_KAFKA_READY_INTERVAL = 3
_READ_TIMEOUT = 15


async def _wait_kafka_accepting() -> None:
    last_exc: Exception | None = None
    for attempt in range(_KAFKA_READY_RETRIES):
        admin = AIOKafkaAdminClient(
            bootstrap_servers=BOOTSTRAP,
            request_timeout_ms=3_000,
        )
        try:
            await admin.start()
            await admin.list_topics()
            await admin.close()
            return
        except Exception as exc:
            last_exc = exc
            try:
                await admin.close()
            except Exception:
                pass
            print(f"Kafka not ready, retry {attempt + 1}/{_KAFKA_READY_RETRIES}...")
            await asyncio.sleep(_KAFKA_READY_INTERVAL)
    raise RuntimeError(
        f"Kafka at {BOOTSTRAP} not accepting connections after "
        f"{_KAFKA_READY_RETRIES * _KAFKA_READY_INTERVAL}s: {last_exc}"
    )


async def _create_topics(topics: list[str]) -> None:
    admin = AIOKafkaAdminClient(bootstrap_servers=BOOTSTRAP)
    await admin.start()
    try:
        new = [NewTopic(name=t, num_partitions=1, replication_factor=1) for t in topics]
        try:
            await admin.create_topics(new)
        except TopicAlreadyExistsError:
            pass
    finally:
        await admin.close()


async def _produce(topic: str, rows: list[dict]) -> None:
    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode(),
    )
    await producer.start()
    try:
        for row in rows:
            await producer.send_and_wait(topic, row)
    finally:
        await producer.stop()


@pytest.fixture(scope="module", autouse=True)
async def kafka_ready():
    await _wait_kafka_accepting()
    await _create_topics(_ALL_TOPICS)


def _make_context(tmp_path: Path) -> PipelineContext:
    store = ArtifactStore(base_path=str(tmp_path / "artifacts"))
    return PipelineContext(artifact_store=store)


def _make_connector(**overrides) -> KafkaConnector:
    return KafkaConnector({**KAFKA_CONFIG, **overrides})


async def _drain_pipeline(pipeline: IngestPipeline, context: PipelineContext) -> list:
    plan = pipeline.plan(context)
    events = []
    async with asyncio.timeout(_READ_TIMEOUT + 5):
        async for event in pipeline.execute(plan, context):
            events.append(event)
    return events


@pytest.fixture
def sample_records(sample_df: pl.DataFrame) -> list[dict]:
    return sample_df.to_dicts()


class TestIngestKafkaValidate:
    def test_valid_config_passes(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(), query={"topic": _TOPIC})
        )
        result = pipeline.validate(_make_context(tmp_path))
        assert result.ok
        assert result.errors == []

    def test_empty_query_fails(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(), query={})
        )
        result = pipeline.validate(_make_context(tmp_path))
        assert not result.ok
        assert any("query" in e for e in result.errors)

    def test_none_connector_fails(self, tmp_path):
        pipeline = IngestPipeline(IngestConfig(connector=None, query={"topic": _TOPIC}))
        result = pipeline.validate(_make_context(tmp_path))
        assert not result.ok
        assert any("connector" in e for e in result.errors)


class TestIngestKafkaPlan:
    def test_plan_has_required_steps(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(), query={"topic": _TOPIC})
        )
        plan = pipeline.plan(_make_context(tmp_path))
        step_names = [s["name"] for s in plan.steps]
        assert "stream_read" in step_names
        assert "materialize_parquet" in step_names
        assert "save_schema" in step_names

    def test_plan_metadata_carries_query(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(), query={"topic": _TOPIC})
        )
        plan = pipeline.plan(_make_context(tmp_path))
        assert plan.metadata["query"]["topic"] == _TOPIC


class TestIngestKafkaExecute:
    @pytest.mark.asyncio
    async def test_event_sequence(self, sample_records, tmp_path):
        await _produce(_TOPIC, sample_records)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="seq-group"),
                query={"topic": _TOPIC},
                options={"max_records": len(sample_records), "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, _make_context(tmp_path))
        types = [e.event_type for e in events]
        assert types[0] == EventType.STARTED
        assert types[-1] == EventType.DONE
        assert EventType.ERROR not in types

    @pytest.mark.asyncio
    async def test_rows_read_matches_source(self, sample_records, sample_df, tmp_path):
        await _produce(_TOPIC, sample_records)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="rows-group"),
                query={"topic": _TOPIC},
                options={"max_records": len(sample_records), "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, _make_context(tmp_path))
        assert events[-1].payload["rows_read"] == len(sample_df)

    @pytest.mark.asyncio
    async def test_parquet_artifact_row_count(self, sample_records, sample_df, tmp_path):
        await _produce(_TOPIC, sample_records)
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="parquet-row-group"),
                query={"topic": _TOPIC},
                options={"max_records": len(sample_records), "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert len(saved_df) == len(sample_df)

    @pytest.mark.asyncio
    async def test_parquet_artifact_columns(self, sample_records, sample_df, tmp_path):
        await _produce(_TOPIC, sample_records)
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="parquet-col-group"),
                query={"topic": _TOPIC},
                options={"max_records": len(sample_records), "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, ctx)
        saved_df = ctx.artifact_store.load_parquet(events[-1].payload["output_artifact_id"])
        assert set(saved_df.columns) == set(sample_df.columns)

    @pytest.mark.asyncio
    async def test_schema_artifact_row_and_column_count(self, sample_records, sample_df, tmp_path):
        await _produce(_TOPIC, sample_records)
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="schema-count-group"),
                query={"topic": _TOPIC},
                options={"max_records": len(sample_records), "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["row_count"] == len(sample_df)
        assert schema_data["column_count"] == len(sample_df.columns)

    @pytest.mark.asyncio
    async def test_schema_artifact_field_names(self, sample_records, sample_df, tmp_path):
        await _produce(_TOPIC, sample_records)
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="schema-names-group"),
                query={"topic": _TOPIC},
                options={"max_records": len(sample_records), "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert set(col["name"] for col in schema_data["schema"]) == set(sample_df.columns)

    @pytest.mark.asyncio
    async def test_schema_artifact_duration_non_negative(self, sample_records, tmp_path):
        await _produce(_TOPIC, sample_records)
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="schema-dur-group"),
                query={"topic": _TOPIC},
                options={"max_records": len(sample_records), "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, ctx)
        schema_data = await ctx.artifact_store.load_json(events[-1].payload["schema_artifact_id"])
        assert schema_data["ingest_duration_s"] >= 0

    @pytest.mark.asyncio
    async def test_progress_events_have_rows_read(self, sample_records, tmp_path):
        await _produce(_TOPIC, sample_records)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="progress-group"),
                query={"topic": _TOPIC},
                options={"max_records": len(sample_records), "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, _make_context(tmp_path))
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress) >= 1
        for e in progress:
            assert e.payload["rows_read"] > 0

    @pytest.mark.asyncio
    async def test_done_event_has_artifact_ids(self, sample_records, tmp_path):
        await _produce(_TOPIC, sample_records)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="artifact-ids-group"),
                query={"topic": _TOPIC},
                options={"max_records": len(sample_records), "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, _make_context(tmp_path))
        assert "output_artifact_id" in events[-1].payload
        assert "schema_artifact_id" in events[-1].payload

    @pytest.mark.asyncio
    async def test_parquet_artifact_is_on_disk(self, sample_records, tmp_path):
        await _produce(_TOPIC, sample_records)
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="disk-parquet-group"),
                query={"topic": _TOPIC},
                options={"max_records": len(sample_records), "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, ctx)
        assert ctx.artifact_store.exists(events[-1].payload["output_artifact_id"], "parquet")

    @pytest.mark.asyncio
    async def test_schema_artifact_is_on_disk(self, sample_records, tmp_path):
        await _produce(_TOPIC, sample_records)
        ctx = _make_context(tmp_path)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="disk-schema-group"),
                query={"topic": _TOPIC},
                options={"max_records": len(sample_records), "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, ctx)
        assert ctx.artifact_store.exists(events[-1].payload["schema_artifact_id"], "json")

    @pytest.mark.asyncio
    async def test_respects_max_records(self, sample_records, tmp_path):
        await _produce(_TOPIC, sample_records)
        max_records = max(1, len(sample_records) // 2)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="max-records-group"),
                query={"topic": _TOPIC},
                options={"max_records": max_records, "timeout_ms": 5_000},
            )
        )
        events = await _drain_pipeline(pipeline, _make_context(tmp_path))
        assert events[-1].payload["rows_read"] == max_records

    @pytest.mark.asyncio
    async def test_custom_batch_size_yields_multiple_progress(self, sample_records, tmp_path):
        await _produce(_TOPIC, sample_records)
        batch_size = max(1, len(sample_records) // 3)
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(group_id="batch-group"),
                query={"topic": _TOPIC},
                options={
                    "max_records": len(sample_records),
                    "batch_size": batch_size,
                    "timeout_ms": 5_000,
                },
            )
        )
        events = await _drain_pipeline(pipeline, _make_context(tmp_path))
        progress = [e for e in events if e.event_type == EventType.PROGRESS]
        assert len(progress) >= 2


class TestIngestKafkaErrorPath:
    @pytest.mark.asyncio
    async def test_bad_broker_raises(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(bootstrap_servers="localhost:19092"),
                query={"topic": _TOPIC},
                options={"max_records": 1, "timeout_ms": 500},
            )
        )
        with pytest.raises(Exception):
            await _drain_pipeline(pipeline, _make_context(tmp_path))

    @pytest.mark.asyncio
    async def test_error_event_emitted_before_raise(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(bootstrap_servers="localhost:19092"),
                query={"topic": _TOPIC},
                options={"max_records": 1, "timeout_ms": 500},
            )
        )
        ctx = _make_context(tmp_path)
        events = []
        try:
            plan = pipeline.plan(ctx)
            async with asyncio.timeout(10):
                async for event in pipeline.execute(plan, ctx):
                    events.append(event)
        except Exception:
            pass
        assert any(e.event_type == EventType.ERROR for e in events)


class TestIngestKafkaSerialize:
    def test_feature_field(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(), query={"topic": _TOPIC})
        )
        assert pipeline.serialize()["feature"] == "ingest"

    def test_query_preserved(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(), query={"topic": _TOPIC})
        )
        assert pipeline.serialize()["query"]["topic"] == _TOPIC

    def test_options_preserved(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(
                connector=_make_connector(),
                query={"topic": _TOPIC},
                options={"max_records": 500},
            )
        )
        assert pipeline.serialize()["options"]["max_records"] == 500

    def test_is_json_serializable(self, tmp_path):
        pipeline = IngestPipeline(
            IngestConfig(connector=_make_connector(), query={"topic": _TOPIC})
        )
        json.dumps(pipeline.serialize())