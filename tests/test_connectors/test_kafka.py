import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch, call

import polars as pl
import pytest

from dataprep.connectors.kafka import (
    KafkaConnector,
    _decode,
    _serialize,
    _bootstrap,
)


SAMPLE_ROWS = [
    {"id": 1, "name": "Alice", "score": 9.5},
    {"id": 2, "name": "Bob",   "score": 8.0},
    {"id": 3, "name": "Carol", "score": 7.5},
]

_BASE_CONFIG = {"bootstrap_servers": "localhost:9092"}


def _make_connector(extra: dict | None = None) -> KafkaConnector:
    return KafkaConnector({**_BASE_CONFIG, **(extra or {})})


def _json_msg(row: dict) -> MagicMock:
    msg = MagicMock()
    msg.value = json.dumps(row).encode()
    return msg


def _make_async_iter(rows: list[dict]):
    async def _gen():
        for r in rows:
            yield _json_msg(r)
    return _gen()


def _patch_consumer(messages: list[dict]):
    consumer = MagicMock()
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    consumer.__aiter__ = MagicMock(return_value=_make_async_iter(messages))

    patcher = patch(
        "dataprep.connectors.kafka.AIOKafkaConsumer",
        return_value=consumer,
    )
    return patcher, consumer


def _patch_producer():
    producer = MagicMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()

    patcher = patch(
        "dataprep.connectors.kafka.AIOKafkaProducer",
        return_value=producer,
    )
    return patcher, producer


class TestDecode:
    def test_json_returns_dict(self):
        raw = json.dumps({"x": 1}).encode()
        assert _decode(raw, "json") == {"x": 1}

    def test_json_invalid_returns_none(self):
        assert _decode(b"not-json{{", "json") is None

    def test_bytes_wraps_in_dict(self):
        result = _decode(b"\x00\x01", "bytes")
        assert result == {"value": b"\x00\x01"}

    def test_csv_returns_first_row(self):
        raw = b"a,b\n1,2\n"
        result = _decode(raw, "csv")
        assert result == {"a": 1, "b": 2}

    def test_unknown_format_does_not_raise_on_bytes(self):
        result = _decode(b"hello", "bytes")
        assert result is not None


class TestSerialize:
    def test_json_round_trip(self):
        row = {"id": 1, "name": "Alice"}
        raw = _serialize(row, "json")
        assert json.loads(raw) == row

    def test_bytes_with_bytes_value(self):
        row = {"value": b"hello"}
        assert _serialize(row, "bytes") == b"hello"

    def test_bytes_with_str_value(self):
        row = {"value": "hello"}
        assert _serialize(row, "bytes") == b"hello"

    def test_csv_produces_decodable_csv(self):
        row = {"a": 1, "b": 2}
        raw = _serialize(row, "csv")
        df = pl.read_csv(raw)
        assert df["a"][0] == 1
        assert df["b"][0] == 2


class TestBootstrap:
    def test_string_passthrough(self):
        assert _bootstrap({"bootstrap_servers": "host:9092"}) == "host:9092"

    def test_list_joined(self):
        result = _bootstrap({"bootstrap_servers": ["a:9092", "b:9092"]})
        assert result == "a:9092,b:9092"


class TestKafkaConnectorRead:
    @pytest.mark.asyncio
    async def test_read_returns_dataframe(self):
        connector = _make_connector()
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        with patcher:
            result = await connector.read({"topic": "test-topic"})
        assert isinstance(result, pl.DataFrame)

    @pytest.mark.asyncio
    async def test_read_correct_row_count(self):
        connector = _make_connector()
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        with patcher:
            result = await connector.read({"topic": "test-topic"})
        assert len(result) == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_read_correct_columns(self):
        connector = _make_connector()
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        with patcher:
            result = await connector.read({"topic": "test-topic"})
        assert set(result.columns) == {"id", "name", "score"}

    @pytest.mark.asyncio
    async def test_read_respects_max_records(self):
        connector = _make_connector()
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        with patcher:
            result = await connector.read(
                {"topic": "test-topic"}, {"max_records": 2}
            )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_read_empty_topic_returns_empty_df(self):
        connector = _make_connector()
        patcher, _ = _patch_consumer([])
        with patcher:
            result = await connector.read({"topic": "empty-topic"})
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_read_skips_unparseable_messages(self):
        connector = _make_connector()

        bad_msg = MagicMock()
        bad_msg.value = b"{{invalid-json}}"

        consumer = MagicMock()
        consumer.start = AsyncMock()
        consumer.stop = AsyncMock()

        async def _gen():
            yield bad_msg
            yield _json_msg(SAMPLE_ROWS[0])

        consumer.__aiter__ = MagicMock(return_value=_gen())

        with patch("dataprep.connectors.kafka.AIOKafkaConsumer", return_value=consumer):
            result = await connector.read({"topic": "t"})

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_read_calls_consumer_start_and_stop(self):
        connector = _make_connector()
        patcher, consumer = _patch_consumer(SAMPLE_ROWS)
        with patcher:
            await connector.read({"topic": "t"})
        consumer.start.assert_awaited_once()
        consumer.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_read_stops_consumer_on_error(self):
        connector = _make_connector()

        consumer = MagicMock()
        consumer.start = AsyncMock()
        consumer.stop = AsyncMock()

        async def _error_gen():
            raise RuntimeError("broker down")
            yield

        consumer.__aiter__ = MagicMock(return_value=_error_gen())

        with patch("dataprep.connectors.kafka.AIOKafkaConsumer", return_value=consumer):
            with pytest.raises(RuntimeError):
                await connector.read({"topic": "t"})

        consumer.stop.assert_awaited_once()


class TestKafkaConnectorStream:
    @pytest.mark.asyncio
    async def test_stream_yields_dataframe_chunks(self):
        connector = _make_connector()
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        chunks = []
        with patcher:
            async for chunk in connector.read_stream(
                {"topic": "t"}, {"batch_size": 2}
            ):
                chunks.append(chunk)
        assert all(isinstance(c, pl.DataFrame) for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_total_rows_match(self):
        connector = _make_connector()
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        total = 0
        with patcher:
            async for chunk in connector.read_stream({"topic": "t"}):
                total += len(chunk)
        assert total == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_stream_batch_size_respected(self):
        connector = _make_connector()
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        sizes = []
        with patcher:
            async for chunk in connector.read_stream(
                {"topic": "t"}, {"batch_size": 2}
            ):
                sizes.append(len(chunk))
        assert sizes == [2, 1]

    @pytest.mark.asyncio
    async def test_stream_max_records_limits_total(self):
        connector = _make_connector()
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        total = 0
        with patcher:
            async for chunk in connector.read_stream(
                {"topic": "t"}, {"max_records": 2, "batch_size": 10}
            ):
                total += len(chunk)
        assert total == 2

    @pytest.mark.asyncio
    async def test_stream_empty_topic_yields_nothing(self):
        connector = _make_connector()
        patcher, _ = _patch_consumer([])
        chunks = []
        with patcher:
            async for chunk in connector.read_stream({"topic": "t"}):
                chunks.append(chunk)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_flushes_remaining_buffer(self):
        connector = _make_connector()
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        chunks = []
        with patcher:
            async for chunk in connector.read_stream(
                {"topic": "t"}, {"batch_size": 10}
            ):
                chunks.append(chunk)
        assert len(chunks) == 1
        assert len(chunks[0]) == 3


class TestKafkaConnectorWrite:
    @pytest.mark.asyncio
    async def test_write_returns_correct_row_count(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS)
        patcher, _ = _patch_producer()
        with patcher:
            result = await connector.write(df, {"topic": "out"})
        assert result.rows_written == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_write_duration_non_negative(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS)
        patcher, _ = _patch_producer()
        with patcher:
            result = await connector.write(df, {"topic": "out"})
        assert result.duration_s >= 0

    @pytest.mark.asyncio
    async def test_write_calls_send_once_per_row(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS)
        patcher, producer = _patch_producer()
        with patcher:
            await connector.write(df, {"topic": "out"})
        assert producer.send_and_wait.await_count == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_write_sends_to_correct_topic(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS[:1])
        patcher, producer = _patch_producer()
        with patcher:
            await connector.write(df, {"topic": "target-topic"})
        topic_arg = producer.send_and_wait.call_args_list[0].args[0]
        assert topic_arg == "target-topic"

    @pytest.mark.asyncio
    async def test_write_with_key_column_sets_key(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS[:1])
        patcher, producer = _patch_producer()
        with patcher:
            await connector.write(
                df, {"topic": "t"}, options={"key_column": "id"}
            )
        kwargs = producer.send_and_wait.call_args_list[0].kwargs
        assert kwargs["key"] == b"1"

    @pytest.mark.asyncio
    async def test_write_without_key_column_key_is_none(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS[:1])
        patcher, producer = _patch_producer()
        with patcher:
            await connector.write(df, {"topic": "t"})
        kwargs = producer.send_and_wait.call_args_list[0].kwargs
        assert kwargs["key"] is None

    @pytest.mark.asyncio
    async def test_write_value_is_bytes(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS[:1])
        patcher, producer = _patch_producer()
        with patcher:
            await connector.write(df, {"topic": "t"})
        kwargs = producer.send_and_wait.call_args_list[0].kwargs
        assert isinstance(kwargs["value"], bytes)

    @pytest.mark.asyncio
    async def test_write_stops_producer_after_finish(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS)
        patcher, producer = _patch_producer()
        with patcher:
            await connector.write(df, {"topic": "t"})
        producer.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_empty_dataframe_sends_nothing(self):
        connector = _make_connector()
        df = pl.DataFrame({"id": [], "name": []})
        patcher, producer = _patch_producer()
        with patcher:
            result = await connector.write(df, {"topic": "t"})
        assert result.rows_written == 0
        producer.send_and_wait.assert_not_awaited()


class TestKafkaConnectorSchema:
    @pytest.mark.asyncio
    async def test_schema_without_default_topic_returns_empty(self):
        connector = _make_connector()
        schema = await connector.schema()
        assert schema.columns == []

    @pytest.mark.asyncio
    async def test_schema_returns_correct_column_count(self):
        connector = _make_connector({"default_topic": "my-topic"})
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        with patcher:
            schema = await connector.schema()
        assert len(schema.columns) == 3

    @pytest.mark.asyncio
    async def test_schema_column_names_match(self):
        connector = _make_connector({"default_topic": "my-topic"})
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        with patcher:
            schema = await connector.schema()
        names = {c.name for c in schema.columns}
        assert names == {"id", "name", "score"}

    @pytest.mark.asyncio
    async def test_schema_all_columns_nullable(self):
        connector = _make_connector({"default_topic": "my-topic"})
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        with patcher:
            schema = await connector.schema()
        assert all(c.nullable for c in schema.columns)

    @pytest.mark.asyncio
    async def test_schema_empty_topic_returns_empty(self):
        connector = _make_connector({"default_topic": "empty-topic"})
        patcher, _ = _patch_consumer([])
        with patcher:
            schema = await connector.schema()
        assert schema.columns == []

    @pytest.mark.asyncio
    async def test_schema_read_error_returns_empty(self):
        connector = _make_connector({"default_topic": "bad-topic"})
        with patch.object(connector, "read", side_effect=Exception("broker down")):
            schema = await connector.schema()
        assert schema.columns == []


class TestKafkaConnectorConnection:
    @pytest.mark.asyncio
    async def test_connection_ok_when_broker_reachable(self):
        connector = _make_connector()
        admin = MagicMock()
        admin.start = AsyncMock()
        admin.close = AsyncMock()
        admin.list_topics = AsyncMock(return_value=["topic-a"])

        with patch("dataprep.connectors.kafka.AIOKafkaAdminClient", return_value=admin):
            status = await connector.test_connection()

        assert status.ok is True
        assert status.latency_ms is not None
        assert status.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_connection_not_ok_when_broker_unreachable(self):
        connector = _make_connector()
        admin = MagicMock()
        admin.start = AsyncMock(side_effect=Exception("Connection refused"))
        admin.close = AsyncMock()

        with patch("dataprep.connectors.kafka.AIOKafkaAdminClient", return_value=admin):
            status = await connector.test_connection()

        assert status.ok is False
        assert status.error is not None

    @pytest.mark.asyncio
    async def test_connection_error_message_propagated(self):
        connector = _make_connector()
        admin = MagicMock()
        admin.start = AsyncMock(side_effect=Exception("ECONNREFUSED"))
        admin.close = AsyncMock()

        with patch("dataprep.connectors.kafka.AIOKafkaAdminClient", return_value=admin):
            status = await connector.test_connection()

        assert "ECONNREFUSED" in status.error

    @pytest.mark.asyncio
    async def test_connection_admin_always_closed(self):
        """admin.close() phải được gọi dù start() raise."""
        connector = _make_connector()
        admin = MagicMock()
        admin.start = AsyncMock(side_effect=Exception("fail"))
        admin.close = AsyncMock()

        with patch("dataprep.connectors.kafka.AIOKafkaAdminClient", return_value=admin):
            await connector.test_connection()

        admin.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connection_latency_positive_on_success(self):
        connector = _make_connector()
        admin = MagicMock()
        admin.start = AsyncMock()
        admin.close = AsyncMock()
        admin.list_topics = AsyncMock(return_value=[])

        with patch("dataprep.connectors.kafka.AIOKafkaAdminClient", return_value=admin):
            status = await connector.test_connection()

        assert status.latency_ms >= 0


class TestKafkaConnectorClose:
    @pytest.mark.asyncio
    async def test_close_does_not_raise(self):
        connector = _make_connector()
        await connector.close()


class TestKafkaConnectorFormats:
    @pytest.mark.asyncio
    async def test_write_json_value_is_valid_json(self):
        connector = _make_connector({"value_format": "json"})
        df = pl.DataFrame(SAMPLE_ROWS[:1])
        patcher, producer = _patch_producer()
        with patcher:
            await connector.write(df, {"topic": "t"})
        raw = producer.send_and_wait.call_args_list[0].kwargs["value"]
        parsed = json.loads(raw)
        assert parsed["id"] == 1

    @pytest.mark.asyncio
    async def test_read_uses_configured_format(self):
        connector = _make_connector({"value_format": "json"})
        patcher, _ = _patch_consumer(SAMPLE_ROWS)
        with patcher:
            result = await connector.read({"topic": "t"})
        assert len(result) == len(SAMPLE_ROWS)