from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Any
import asyncio
import io
import asyncpg
import polars as pl

from ..base import BaseConnector, ConnectionStatus
from ...utils.connection import timed_connect


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
    fetch_size: int = 200_000
    server_settings: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.fetch_size <= 0:
            self.fetch_size = 200_000


class PostgreSqlConnector(BaseConnector[PostgreSQLConnectorConfig]):
    def __init__(self, config: PostgreSQLConnectorConfig):
        super().__init__(config)
        self.connection_string = (
            f"postgresql://{config.user}:{config.password}"
            f"@{config.host}:{config.port}/{config.database}"
        )
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> ConnectionStatus:
        async def probe():
            self.pool = await asyncpg.create_pool(
                dsn=self.connection_string,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                ssl=self.config.ssl,
                timeout=self.config.connect_timeout,
                command_timeout=self.config.command_timeout,
                statement_cache_size=self.config.statement_cache_size,
                server_settings=self.config.server_settings,
            )
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

        ok, latency_ms, error = await timed_connect(probe)
        return ConnectionStatus(ok=ok, latency_ms=latency_ms, error=error)

    async def cleanup(self):
        if self.pool:
            await self.pool.close()
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
            buf = io.BytesIO()
            async with self.pool.acquire() as conn:
                await conn.copy_from_query(
                    sql,
                    **({"args": params} if params else {}),
                    output=buf,
                    format="csv",
                    header=True,
                )
            buf.seek(0)
            return pl.read_csv(buf)

        async def generate() -> AsyncGenerator[pl.DataFrame, Any]:
            nb = batch_size if batch_size and batch_size > 0 else self.config.fetch_size
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    cur = await conn.cursor(sql, *(params or []))
                    while True:
                        rows = await cur.fetch(nb)
                        if not rows:
                            break
                        keys = list(rows[0].keys())
                        yield pl.DataFrame(
                            {k: [r[k] for r in rows] for k in keys}
                        )

        return generate()

    async def execute(
        self,
        sql: str,
        params: Optional[list] = None,
        many: bool = False,
    ) -> int:
        if not self.pool:
            raise ValueError("Pool not connected")
        async with self.pool.acquire() as conn:
            if many:
                await conn.executemany(sql, params or [])
                return len(params or [])
            result = await conn.execute(sql, *(params or []))
            try:
                return int(result.split()[-1])
            except (ValueError, IndexError):
                return 0

    async def list_tables(self, schema: Optional[str] = None) -> list[str]:
        if not self.pool:
            raise ValueError("Pool not connected")
        s = schema or self.config.schema
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = $1 AND table_type = 'BASE TABLE' "
                "ORDER BY table_name",
                s,
            )
        return [r["table_name"] for r in rows]