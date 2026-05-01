from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Any
import asyncio
import asyncmy
import aiomysql
import connectorx as cx
import polars as pl

from .base import BaseConnector, ConnectionStatus
from ..utils.connection import timed_connect


@dataclass
class MySQLConnectorConfig:
    host: str
    database: str
    user: str
    password: str
    port: int = 3306
    read_only: bool = False
    min_pool_size: int = 1
    max_pool_size: int = 10
    connect_timeout: float = 10.0
    charset: str = "utf8mb4"
    fetch_size: int = 50_000


class MySQLConnector(BaseConnector[MySQLConnectorConfig]):
    def __init__(self, config: MySQLConnectorConfig):
        super().__init__(config)
        self._cx_url = (
            f"mysql://{config.user}:{config.password}"
            f"@{config.host}:{config.port}/{config.database}"
        )
        self._asyncmy_cfg = dict(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            db=config.database,
            charset=config.charset,
            autocommit=True,
        )
        self._aiomysql_cfg = dict(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            db=config.database,
            charset=config.charset,
            autocommit=True,
        )
        self._asyncmy_pool: Optional[asyncmy.Pool] = None
        self._aiomysql_pool: Optional[aiomysql.Pool] = None

    async def connect(self) -> ConnectionStatus:
        async def probe():
            self._asyncmy_pool = await asyncmy.create_pool(
                **self._asyncmy_cfg,
                minsize=self.config.min_pool_size,
                maxsize=self.config.max_pool_size,
                connect_timeout=self.config.connect_timeout,
            )
            self._aiomysql_pool = await aiomysql.create_pool(
                **self._aiomysql_cfg,
                minsize=self.config.min_pool_size,
                maxsize=self.config.max_pool_size,
                connect_timeout=self.config.connect_timeout,
            )
            async with self._asyncmy_pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")

        ok, latency_ms, error = await timed_connect(probe)
        return ConnectionStatus(ok=ok, latency_ms=latency_ms, error=error)

    async def cleanup(self):
        if self._asyncmy_pool:
            self._asyncmy_pool.close()
            await self._asyncmy_pool.wait_closed()
            self._asyncmy_pool = None
        if self._aiomysql_pool:
            self._aiomysql_pool.close()
            await self._aiomysql_pool.wait_closed()
            self._aiomysql_pool = None

    async def query(
        self,
        sql: str,
        params: Optional[list] = None,
        stream: bool = False,
        batch_size: Optional[int] = None,
    ) -> pl.DataFrame | AsyncGenerator[pl.DataFrame, Any]:
        if not self._asyncmy_pool or not self._aiomysql_pool:
            raise ValueError("Pool not connected")

        if not stream:
            return await asyncio.to_thread(
                cx.read_sql, self._cx_url, sql, return_type="polars"
            )

        async def _stream_gen() -> AsyncGenerator[pl.DataFrame, Any]:
            nb = batch_size or self.config.fetch_size
            async with self._aiomysql_pool.acquire() as conn:
                async with conn.cursor(aiomysql.SSCursor) as cur:
                    await cur.execute(sql, params or None)
                    cols = [d[0] for d in cur.description]
                    n_cols = len(cols)
                    while True:
                        rows = await cur.fetchmany(nb)
                        if not rows:
                            break
                        yield pl.DataFrame(
                            {cols[i]: [row[i] for row in rows] for i in range(n_cols)}
                        )

        return _stream_gen()

    async def execute(
        self,
        sql: str,
        params: Optional[list] = None,
        many: bool = False,
    ) -> int:
        if not self._asyncmy_pool:
            raise ValueError("Pool not connected")
        async with self._asyncmy_pool.acquire() as conn:
            async with conn.cursor() as cur:
                if many:
                    await cur.executemany(sql, params or [])
                    return len(params or [])
                await cur.execute(sql, params or [])
                return cur.rowcount

    async def list_tables(self, schema: Optional[str] = None) -> list[str]:
        if not self._asyncmy_pool:
            raise ValueError("Pool not connected")
        db = schema or self.config.database
        async with self._asyncmy_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_type = 'BASE TABLE' "
                    "ORDER BY table_name",
                    (db,),
                )
                rows = await cur.fetchall()
        return [r[0] for r in rows]