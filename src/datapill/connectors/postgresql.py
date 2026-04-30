import time
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Any
import polars as pl
import asyncpg

from .base import BaseConnector, ConnectionStatus
from ..utils.streaming import estimate_from_sample, records_to_df

@dataclass
class PostgreSQLConnectorConfig:
    host: str
    database: str
    user: str
    password: str
    port: int = 5432
    schema: str = "public"
    read_only: bool = False
    min_pool_size: int = 1
    max_pool_size: int = 10
    ssl: Optional[str] = None
    connect_timeout: float = 10.0
    command_timeout: float = 60.0
    statement_cache_size: int = 100
    server_settings: dict = field(default_factory=dict)


class PostgreSqlConnector(BaseConnector[PostgreSQLConnectorConfig]):
    def __init__(self, config: PostgreSQLConnectorConfig):
        super().__init__(config)
        self.connection_string = f"postgresql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}"
        self.pool = None

    async def connect(self) -> ConnectionStatus:
        t0 = time.perf_counter()

        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.connection_string,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                ssl=self.config.ssl,
                connect_timeout=self.config.connect_timeout,
                command_timeout=self.config.command_timeout,
                statement_cache_size=self.config.statement_cache_size,
                server_settings=self.config.server_settings,
            )

            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            return ConnectionStatus(ok=True, latency_ms=1000 * (time.perf_counter() - t0))

        except Exception as e:
            return ConnectionStatus(ok=False, error=str(e))


    async def cleanup(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def query(self, sql: str, params: Optional[list], stream: bool = False, batch_size: Optional[int] = None) -> pl.DataFrame | AsyncGenerator[pl.DataFrame, Any]:
        if not self.pool:
            raise ValueError("Pool not connected")

        if not stream:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, *params or [])

            if not rows:
                return pl.DataFrame()

            return pl.DataFrame([dict(r) for r in rows])

    async def query(
        self,
        sql: str,
        params: Optional[list],
        stream: bool = False,
        batch_size: Optional[int] = None,
    ) -> pl.DataFrame | AsyncGenerator[pl.DataFrame, Any]:
        if not self.pool:
            raise ValueError("Pool not connected")

        if not stream:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, *params or [])
            return records_to_df(rows)

        async def _stream() -> AsyncGenerator[pl.DataFrame, Any]:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    cursor = await conn.cursor(sql, *params or [])

                    sample = await cursor.fetch(1000)
                    if not sample:
                        return

                    resolved = estimate_from_sample(sample, batch_size)

                    yield records_to_df(sample)

                    while True:
                        rows = await cursor.fetch(resolved)
                        if not rows:
                            break
                        yield records_to_df(rows)

        return _stream()