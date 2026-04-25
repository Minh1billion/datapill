import asyncio
import json
import pytest
import polars as pl
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError

from dataprep.connectors.kafka import KafkaConnector

BOOTSTRAP = "localhost:9092"

KAFKA_CONFIG = {
    "bootstrap_servers": BOOTSTRAP,
    "group_id": "test-group",
    "auto_offset_reset": "earliest",
    "value_format": "json",
    "default_topic": "test-topic",
}

TOPIC = "test-topic"
WRITE_TOPIC = "write-topic"
EMPTY_TOPIC = "empty-topic"
ALL_TOPICS = [TOPIC, WRITE_TOPIC, EMPTY_TOPIC]

_READ_TIMEOUT = 15


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


async def _drain(topic: str, expected: int, group: str = "drain-group") -> list[dict]:
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=BOOTSTRAP,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode()),
        group_id=group,
    )
    await consumer.start()
    records = []
    try:
        async with asyncio.timeout(_READ_TIMEOUT):
            async for msg in consumer:
                records.append(msg.value)
                if len(records) >= expected:
                    break
    finally:
        await consumer.stop()
    return records


async def _read_with_timeout(connector: KafkaConnector, query: dict, options: dict) -> pl.DataFrame:
    timeout_s = options.get("timeout_ms", 5_000) / 1000 + _READ_TIMEOUT
    async with asyncio.timeout(timeout_s):
        return await connector.read(query, options)


async def _stream_with_timeout(connector: KafkaConnector, query: dict, options: dict) -> list[pl.DataFrame]:
    frames = []
    timeout_s = options.get("timeout_ms", 5_000) / 1000 + _READ_TIMEOUT
    async with asyncio.timeout(timeout_s):
        async for chunk in connector.read_stream(query, options):
            frames.append(chunk)
    return frames


@pytest.fixture(scope="module", autouse=True)
async def kafka_topics():
    await _create_topics(ALL_TOPICS)


@pytest.fixture
def connector():
    return KafkaConnector(KAFKA_CONFIG)


@pytest.fixture
def sample_records(sample_df: pl.DataFrame) -> list[dict]:
    return sample_df.to_dicts()


@pytest.mark.integration
async def test_test_connection(connector: KafkaConnector):
    status = await connector.test_connection()
    assert status.ok
    assert status.latency_ms is not None


@pytest.mark.integration
async def test_read_returns_produced_messages(connector: KafkaConnector, sample_records: list[dict], sample_df: pl.DataFrame):
    await _produce(TOPIC, sample_records)

    df = await _read_with_timeout(
        connector,
        {"topic": TOPIC},
        {"max_records": len(sample_records), "timeout_ms": 5_000},
    )
    assert len(df) == len(sample_df)
    assert set(df.columns) == set(sample_df.columns)


@pytest.mark.integration
async def test_read_respects_max_records(connector: KafkaConnector, sample_records: list[dict]):
    await _produce(TOPIC, sample_records)

    max_records = min(3, len(sample_records))
    df = await _read_with_timeout(
        connector,
        {"topic": TOPIC},
        {"max_records": max_records, "timeout_ms": 5_000},
    )
    assert len(df) == max_records


@pytest.mark.integration
async def test_read_stream_yields_batches(connector: KafkaConnector, sample_records: list[dict], sample_df: pl.DataFrame):
    await _produce(TOPIC, sample_records)

    batch_size = max(1, len(sample_records) // 3)
    frames = await _stream_with_timeout(
        connector,
        {"topic": TOPIC},
        {"batch_size": batch_size, "max_records": len(sample_records), "timeout_ms": 5_000},
    )
    assert len(frames) >= 2
    assert sum(len(f) for f in frames) == len(sample_df)


@pytest.mark.integration
async def test_read_stream_respects_max_records(connector: KafkaConnector, sample_records: list[dict]):
    await _produce(TOPIC, sample_records)

    max_records = min(4, len(sample_records))
    batch_size = max(1, max_records // 2)
    frames = await _stream_with_timeout(
        connector,
        {"topic": TOPIC},
        {"batch_size": batch_size, "max_records": max_records, "timeout_ms": 5_000},
    )
    assert sum(len(f) for f in frames) == max_records


@pytest.mark.integration
async def test_write_produces_messages(connector: KafkaConnector, sample_df: pl.DataFrame):
    result = await connector.write(sample_df, {"topic": WRITE_TOPIC})
    assert result.rows_written == len(sample_df)
    assert result.duration_s >= 0

    records = await _drain(WRITE_TOPIC, len(sample_df), group="drain-write-group")
    assert len(records) == len(sample_df)


@pytest.mark.integration
async def test_write_with_key_column(connector: KafkaConnector, sample_df: pl.DataFrame):
    first_col = sample_df.columns[0]
    result = await connector.write(
        sample_df,
        {"topic": WRITE_TOPIC},
        {"key_column": first_col},
    )
    assert result.rows_written == len(sample_df)


@pytest.mark.integration
async def test_schema_from_default_topic(connector: KafkaConnector, sample_records: list[dict], sample_df: pl.DataFrame):
    await _produce(KAFKA_CONFIG["default_topic"], sample_records)

    async with asyncio.timeout(20):
        info = await connector.schema()

    if info.columns:
        names = [c.name for c in info.columns]
        assert set(names) == set(sample_df.columns)


@pytest.mark.integration
async def test_connection_failure():
    bad = KafkaConnector({**KAFKA_CONFIG, "bootstrap_servers": "localhost:19092"})
    status = await bad.test_connection()
    assert not status.ok
    assert status.error is not None


@pytest.mark.integration
async def test_read_empty_topic_returns_empty_df():
    connector = KafkaConnector({**KAFKA_CONFIG, "group_id": "empty-test-group"})
    with pytest.raises(TimeoutError):
        await _read_with_timeout(
            connector,
            {"topic": EMPTY_TOPIC},
            {"max_records": 1, "timeout_ms": 500},
        )