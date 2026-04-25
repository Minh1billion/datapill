import io
import json
import time
from typing import AsyncGenerator, Any

import polars as pl
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import (
    KafkaConnectionError,
    KafkaError,
    TopicAlreadyExistsError,
    UnknownTopicOrPartitionError,
)

from .base import BaseConnector, ColumnMeta, ConnectionStatus, SchemaInfo, WriteResult

_DEFAULT_GROUP = "dataprep"
_DEFAULT_OFFSET = "earliest"
_CONNECT_TIMEOUT = 10.0
_DEFAULT_MAX_RECORDS = 1_000
_DEFAULT_BATCH_RECORDS = 500


def _bootstrap(config: dict[str, Any]) -> str:
    servers = config["bootstrap_servers"]
    if isinstance(servers, list):
        return ",".join(servers)
    return servers


def _consumer_kwargs(config: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "bootstrap_servers": _bootstrap(config),
        "group_id": config.get("group_id", _DEFAULT_GROUP),
        "auto_offset_reset": config.get("auto_offset_reset", _DEFAULT_OFFSET),
        "enable_auto_commit": False,
        "value_deserializer": lambda v: v,
    }
    _apply_security(kw, config)
    if extra:
        kw.update(extra)
    return kw


def _producer_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "bootstrap_servers": _bootstrap(config),
        "value_serializer": lambda v: v,
    }
    _apply_security(kw, config)
    return kw


def _apply_security(kw: dict, config: dict) -> None:
    protocol = config.get("security_protocol", "PLAINTEXT").upper()
    kw["security_protocol"] = protocol
    if protocol in ("SASL_PLAINTEXT", "SASL_SSL"):
        kw["sasl_mechanism"] = config.get("sasl_mechanism", "PLAIN")
        kw["sasl_plain_username"] = config.get("sasl_username", "")
        kw["sasl_plain_password"] = config.get("sasl_password", "")
    if protocol in ("SSL", "SASL_SSL"):
        if ssl_ca := config.get("ssl_cafile"):
            kw["ssl_cafile"] = ssl_ca


def _decode(raw: bytes, fmt: str) -> dict[str, Any] | None:
    """Decode một Kafka message value → dict.  Trả về None nếu không parse được."""
    try:
        if fmt == "json":
            return json.loads(raw.decode("utf-8"))
        if fmt == "csv":
            return pl.read_csv(io.BytesIO(raw)).to_dicts()[0]
        return {"value": raw}
    except Exception:
        return None


def _records_to_df(records: list[dict[str, Any]]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame()
    return pl.DataFrame(records)


def _serialize(row: dict[str, Any], fmt: str) -> bytes:
    if fmt == "json":
        return json.dumps(row, default=str).encode("utf-8")
    if fmt == "csv":
        df = pl.DataFrame([row])
        buf = io.BytesIO()
        df.write_csv(buf)
        return buf.getvalue()

    v = row.get("value", b"")
    return v if isinstance(v, bytes) else str(v).encode()


class KafkaConnector(BaseConnector):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._fmt: str = config.get("value_format", "json")

    def _make_consumer(self, topic: str, extra: dict | None = None) -> AIOKafkaConsumer:
        return AIOKafkaConsumer(topic, **_consumer_kwargs(self.config, extra))

    def _make_producer(self) -> AIOKafkaProducer:
        return AIOKafkaProducer(**_producer_kwargs(self.config))


    async def read(
        self, query: dict[str, Any], options: dict[str, Any] | None = None
    ) -> pl.DataFrame:
        opts = options or {}
        topic: str = query["topic"]
        max_records: int = opts.get("max_records", _DEFAULT_MAX_RECORDS)
        timeout_ms: int = opts.get("timeout_ms", 3_000)

        consumer = self._make_consumer(topic)
        await consumer.start()
        records: list[dict[str, Any]] = []
        try:
            async for msg in consumer:
                decoded = _decode(msg.value, self._fmt)
                if decoded is not None:
                    records.append(decoded)
                if len(records) >= max_records:
                    break
        except Exception as exc:
            raise RuntimeError(f"Kafka read error: {exc}") from exc
        finally:
            await consumer.stop()

        return _records_to_df(records)


    async def read_stream(
        self, query: dict[str, Any], options: dict[str, Any] | None = None
    ) -> AsyncGenerator[pl.DataFrame, None]:
        opts = options or {}
        topic: str = query["topic"]
        batch_size: int = opts.get("batch_size", _DEFAULT_BATCH_RECORDS)
        max_records: int | None = opts.get("max_records", None)
        timeout_ms: int = opts.get("timeout_ms", 3_000)

        consumer = self._make_consumer(topic)
        await consumer.start()
        buffer: list[dict[str, Any]] = []
        total = 0
        try:
            async for msg in consumer:
                decoded = _decode(msg.value, self._fmt)
                if decoded is not None:
                    buffer.append(decoded)
                    total += 1
                if len(buffer) >= batch_size:
                    yield _records_to_df(buffer)
                    buffer = []
                if max_records is not None and total >= max_records:
                    break
        finally:
            if buffer:
                yield _records_to_df(buffer)
            await consumer.stop()


    async def schema(self) -> SchemaInfo:
        topic = self.config.get("default_topic")
        if not topic:
            return SchemaInfo(columns=[])

        try:
            df = await self.read({"topic": topic}, {"max_records": 1, "timeout_ms": 5_000})
        except Exception:
            return SchemaInfo(columns=[])

        if df.is_empty():
            return SchemaInfo(columns=[])

        return SchemaInfo(columns=[
            ColumnMeta(name=col, dtype=str(dtype), nullable=True)
            for col, dtype in zip(df.columns, df.dtypes)
        ])


    async def test_connection(self) -> ConnectionStatus:
        t0 = time.perf_counter()
        admin = AIOKafkaAdminClient(
            bootstrap_servers=_bootstrap(self.config),
            request_timeout_ms=int(_CONNECT_TIMEOUT * 1000),
        )
        try:
            await admin.start()
            await admin.list_topics()
            return ConnectionStatus(ok=True, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return ConnectionStatus(ok=False, error=str(exc))
        finally:
            try:
                await admin.close()
            except Exception:
                pass


    async def write(
        self,
        df: pl.DataFrame,
        target: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> WriteResult:
        opts = options or {}
        topic: str = target["topic"]
        key_col: str | None = opts.get("key_column")
        fmt: str = opts.get("value_format", self._fmt)

        t0 = time.perf_counter()
        producer = self._make_producer()
        await producer.start()
        rows_written = 0
        try:
            for row in df.iter_rows(named=True):
                value = _serialize(row, fmt)
                key: bytes | None = None
                if key_col and key_col in row:
                    key = str(row[key_col]).encode()
                await producer.send_and_wait(topic, value=value, key=key)
                rows_written += 1
        finally:
            await producer.stop()

        return WriteResult(rows_written=rows_written, duration_s=time.perf_counter() - t0)


    async def close(self) -> None:
        pass