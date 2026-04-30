import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Any
import polars as pl
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError

from .base import BaseConnector, ConnectionStatus


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
        self._auth = dict(
            security_protocol=config.security_protocol,
            **(
                dict(
                    sasl_mechanism=config.sasl_mechanism,
                    sasl_plain_username=config.sasl_username,
                    sasl_plain_password=config.sasl_password,
                )
                if config.sasl_mechanism else {}
            ),
        )

    async def connect(self) -> ConnectionStatus:
        import time
        t0 = time.perf_counter()
        try:
            consumer = AIOKafkaConsumer(
                bootstrap_servers=self._bootstrap,
                request_timeout_ms=self.config.request_timeout_ms,
                **self._auth,
            )
            await consumer.start()
            await consumer.topics()
            await consumer.stop()
            return ConnectionStatus(ok=True, latency_ms=1000 * (time.perf_counter() - t0))
        except Exception as e:
            return ConnectionStatus(ok=False, error=str(e))

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
        consumer = AIOKafkaConsumer(
            bootstrap_servers=self._bootstrap,
            **self._auth,
        )
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