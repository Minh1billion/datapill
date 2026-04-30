import time
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Any
import polars as pl
import asyncmy
import asyncmy.cursors

from .base import BaseConnector, ConnectionStatus
from ..utils.streaming import estimate_from_sample, records_to_df


@dataclass
class MySQLConnectorConfig:
    host: str
    database: str
    user: str
    password: str
    port: int = 3306
    schema: str = None
    read_only: bool = False
    min_pool_size: int = 1
    max_pool_size: int = 10
    connect_timeout: float = 10.0
    charset: str = "utf8mb4"
    server_settings: dict = field(default_factory=dict)


class MySQLConnector(BaseConnector[MySQLConnectorConfig]):
    def __init__(self, config: MySQLConnectorConfig):
        super().__init__(config)
        self.pool = None

    async def connect(self) -> ConnectionStatus:
        t0 = time.perf_counter()
        try:
            self.pool = await asyncmy.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                db=self.config.database,
                minsize=self.config.min_pool_size,
                maxsize=self.config.max_pool_size,
                connect_timeout=self.config.connect_timeout,
                charset=self.config.charset,
                autocommit=True,
            )
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
            return ConnectionStatus(ok=True, latency_ms=1000 * (time.perf_counter() - t0))
        except Exception as e:
            return ConnectionStatus(ok=False, error=str(e))

    async def cleanup(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self.pool = None

    async def query(
        self,
        sql: str,
        params: Optional[list] = None,
        stream: bool = False,
        batch_size: Optional[int] = None,
    ) -> pl.DataFrame | AsyncGenerator[pl.DataFrame, Any]:
        if not self.pool:
            raise ValueError("Pool not connected")

        if not stream:
            async with self.pool.acquire() as conn:
                async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
                    await cur.execute(sql, params or [])
                    rows = await cur.fetchall()
            if not rows:
                return pl.DataFrame()
            return pl.DataFrame({k: [r[k] for r in rows] for k in rows[0]})

        async def _stream() -> AsyncGenerator[pl.DataFrame, Any]:
            async with self.pool.acquire() as conn:
                async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
                    await cur.execute(sql, params or [])
                    sample = await cur.fetchmany(1000)
                    if not sample:
                        return
                    resolved = estimate_from_sample(sample, batch_size)
                    yield records_to_df(sample)
                    while True:
                        rows = await cur.fetchmany(resolved)
                        if not rows:
                            break
                        yield records_to_df(rows)

        return _stream()

    async def execute(
        self,
        sql: str,
        params: Optional[list] = None,
        many: bool = False,
    ) -> int:
        if not self.pool:
            raise ValueError("Pool not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                if many:
                    await cur.executemany(sql, params or [])
                    return len(params or [])
                await cur.execute(sql, params or [])
                return cur.rowcount

    async def list_tables(self, schema: Optional[str] = None) -> list[str]:
        if not self.pool:
            raise ValueError("Pool not connected")
        db = schema or self.config.database
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_type = 'BASE TABLE' "
                    "ORDER BY table_name",
                    (db,),
                )
                rows = await cur.fetchall()
        return [r[0] for r in rows]