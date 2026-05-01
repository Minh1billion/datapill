import json
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Any
import polars as pl
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from .base import BaseConnector, ConnectionStatus
from ..utils.connection import timed_connect
from ..utils.auth import build_sasl_auth


@dataclass
class KafkaConnectorConfig:
    brokers: list[str]
    group_id: str = "datapill"
    auto_offset_reset: str = "earliest"
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    request_timeout_ms: int = 40000
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    producer_acks: str = "all"
    extra: dict = field(default_factory=dict)


class KafkaConnector(BaseConnector[KafkaConnectorConfig]):
    def __init__(self, config: KafkaConnectorConfig):
        super().__init__(config)
        self._bootstrap = ",".join(config.brokers)
        self._auth = {"security_protocol": config.security_protocol}
        if config.sasl_mechanism:
            self._auth.update(build_sasl_auth(config.sasl_mechanism, config.sasl_username, config.sasl_password))

    async def connect(self) -> ConnectionStatus:
        async def probe():
            consumer = AIOKafkaConsumer(
                bootstrap_servers=self._bootstrap,
                request_timeout_ms=self.config.request_timeout_ms,
                **self._auth,
            )
            await consumer.start()
            await consumer.topics()
            await consumer.stop()

        ok, latency_ms, error = await timed_connect(probe)
        return ConnectionStatus(ok=ok, latency_ms=latency_ms, error=error)

    async def cleanup(self) -> None:
        return

    async def consume(
        self,
        topic: str,
        max_messages: Optional[int] = None,
        timeout_ms: int = 5000,
    ) -> AsyncGenerator[pl.DataFrame, Any]:
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self._bootstrap,
            group_id=self.config.group_id,
            auto_offset_reset=self.config.auto_offset_reset,
            max_poll_records=self.config.max_poll_records,
            session_timeout_ms=self.config.session_timeout_ms,
            **self._auth,
        )
        await consumer.start()
        received = 0
        try:
            while True:
                batch = await consumer.getmany(timeout_ms=timeout_ms, max_records=self.config.max_poll_records)
                if not batch:
                    break
                records = []
                for tp, messages in batch.items():
                    for msg in messages:
                        try:
                            records.append(json.loads(msg.value))
                        except Exception:
                            records.append({"_raw": msg.value.decode("utf-8", errors="replace")})
                        received += 1
                if records:
                    yield pl.DataFrame(records)
                if max_messages and received >= max_messages:
                    break
        finally:
            await consumer.stop()

    async def produce(
        self,
        topic: str,
        df: pl.DataFrame,
        key_column: Optional[str] = None,
    ) -> int:
        producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap,
            acks=self.config.producer_acks,
            **self._auth,
        )
        await producer.start()
        sent = 0
        try:
            for row in df.iter_rows(named=True):
                key = str(row[key_column]).encode() if key_column else None
                value = json.dumps(row, default=str).encode()
                await producer.send(topic, value=value, key=key)
                sent += 1
            await producer.flush()
        finally:
            await producer.stop()
        return sent

    async def list_topics(self) -> list[str]:
        consumer = AIOKafkaConsumer(bootstrap_servers=self._bootstrap, **self._auth)
        await consumer.start()
        try:
            topics = await consumer.topics()
        finally:
            await consumer.stop()
        return sorted(topics)

    async def seek(self, topic: str, partition: int, offset: int) -> None:
        from aiokafka import TopicPartition
        consumer = AIOKafkaConsumer(
            bootstrap_servers=self._bootstrap,
            group_id=self.config.group_id,
            **self._auth,
        )
        await consumer.start()
        try:
            tp = TopicPartition(topic, partition)
            consumer.seek(tp, offset)
        finally:
            await consumer.stop()