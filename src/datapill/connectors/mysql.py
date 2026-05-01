from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Any
import asyncio
import asyncmy
import connectorx as cx
import polars as pl

from .base import BaseConnector, ConnectionStatus
from ..utils.connection import timed_connect
from ..utils.streaming import estimate_batch_size


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
    fetch_size: int = 50_000
    server_settings: dict = field(default_factory=dict)


class MySQLConnector(BaseConnector[MySQLConnectorConfig]):
    def __init__(self, config: MySQLConnectorConfig):
        super().__init__(config)
        self.pool = None
        self._cx_url = (
            f"mysql://{config.user}:{config.password}"
            f"@{config.host}:{config.port}/{config.database}"
        )

    async def connect(self) -> ConnectionStatus:
        async def probe():
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
                init_command="SET SESSION net_read_timeout=3600, net_write_timeout=3600",
            )
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")

        ok, latency_ms, error = await timed_connect(probe)
        return ConnectionStatus(ok=ok, latency_ms=latency_ms, error=error)

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
            return await asyncio.to_thread(
                cx.read_sql, self._cx_url, sql, return_type="polars"
            )

        async def _gen():
            df = await asyncio.to_thread(
                cx.read_sql, self._cx_url, sql, return_type="polars"
            )
            nb = batch_size or estimate_batch_size(df.estimated_size(), len(df))
            for i in range(0, len(df), nb):
                yield df.slice(i, nb)

        return _gen()

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