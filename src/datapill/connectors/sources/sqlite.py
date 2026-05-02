from dataclasses import dataclass
from typing import Optional, AsyncGenerator, Any
import asyncio
import aiosqlite
import duckdb
import polars as pl

from ..base import BaseConnector, ConnectionStatus
from ...utils.connection import timed_connect


@dataclass
class SQLiteConnectorConfig:
    path: str
    read_only: bool = False
    fetch_size: int = 50_000
    timeout: float = 30.0
    check_same_thread: bool = False
    cached_statements: int = 128


class SQLiteConnector(BaseConnector[SQLiteConnectorConfig]):
    def __init__(self, config: SQLiteConnectorConfig):
        super().__init__(config)
        self.conn: Optional[aiosqlite.Connection] = None
        self._attach = (
            f"ATTACH '{config.path}' AS _src "
            f"(TYPE sqlite{', READ_ONLY' if config.read_only else ''})"
        )

    async def connect(self) -> ConnectionStatus:
        async def probe():
            self.conn = await aiosqlite.connect(
                self.config.path,
                timeout=self.config.timeout,
                check_same_thread=self.config.check_same_thread,
                cached_statements=self.config.cached_statements,
            )
            self.conn.row_factory = None
            await self.conn.execute("PRAGMA journal_mode=WAL")
            await self.conn.execute("PRAGMA synchronous=NORMAL")
            await self.conn.execute("PRAGMA cache_size=-65536")
            await self.conn.execute("PRAGMA temp_store=MEMORY")
            await self.conn.execute("SELECT 1")

        ok, latency_ms, error = await timed_connect(probe)
        return ConnectionStatus(ok=ok, latency_ms=latency_ms, error=error)

    async def cleanup(self):
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def query(
        self,
        sql: str,
        params: Optional[list] = None,
        stream: bool = False,
        batch_size: Optional[int] = None,
    ) -> "pl.DataFrame | AsyncGenerator[pl.DataFrame, Any]":
        if not self.conn:
            raise ValueError("Not connected")

        if not stream:
            return await asyncio.to_thread(
                lambda: [c := duckdb.connect(), c.execute(self._attach), c.execute("SET search_path='_src'"), c.sql(sql).pl()][-1]
            )

        nb = batch_size or self.config.fetch_size

        async def generate() -> AsyncGenerator[pl.DataFrame, Any]:
            reader = await asyncio.to_thread(
                lambda: [c := duckdb.connect(), c.execute(self._attach), c.execute("SET search_path='_src'"), c.sql(sql).fetch_arrow_reader(nb)][-1]
            )
            for chunk in reader:
                yield pl.from_arrow(chunk)
                await asyncio.sleep(0)

        return generate()

    async def execute(
        self,
        sql: str,
        params: Optional[list] = None,
        many: bool = False,
    ) -> int:
        if not self.conn:
            raise ValueError("Not connected")
        if many:
            await self.conn.executemany(sql, params or [])
            await self.conn.commit()
            return len(params or [])
        cur = await self.conn.execute(sql, params or [])
        await self.conn.commit()
        return cur.rowcount

    async def list_tables(self, schema: Optional[str] = None) -> list[str]:
        if not self.conn:
            raise ValueError("Not connected")
        cur = await self.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        rows = await cur.fetchall()
        return [r[0] for r in rows]