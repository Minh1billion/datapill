import time
from typing import AsyncGenerator, Any
import polars as pl
import asyncpg
from .base import BaseConnector, SchemaInfo, ColumnMeta, ConnectionStatus, WriteResult

_PG_TO_POLARS: dict[str, str] = {
    "integer": "Int32", "bigint": "Int64", "smallint": "Int16",
    "real": "Float32", "double precision": "Float64", "numeric": "Float64", "decimal": "Float64",
    "boolean": "Boolean",
    "character varying": "Utf8", "varchar": "Utf8", "text": "Utf8", "char": "Utf8", "uuid": "Utf8",
    "date": "Date", "timestamp without time zone": "Datetime", "timestamp with time zone": "Datetime",
    "json": "Utf8", "jsonb": "Utf8",
    "bytea": "Binary",
}

_MAX_RETRIES = 3
_BASE_BACKOFF = 0.5


async def _with_retry(coro_factory, max_retries: int = _MAX_RETRIES):
    import asyncio, random
    last_exc = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except (asyncpg.TooManyConnectionsError, asyncpg.PostgresConnectionError, OSError) as exc:
            last_exc = exc
            wait = _BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.3)
            await asyncio.sleep(wait)
    raise last_exc


class PostgreSQLConnector(BaseConnector):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None or self._pool._closed:
            self._pool = await asyncpg.create_pool(
                host=self.config["host"],
                port=self.config.get("port", 5432),
                database=self.config["database"],
                user=self.config["user"],
                password=self.config["password"],
                min_size=1,
                max_size=self.config.get("max_connections", 5),
                command_timeout=self.config.get("timeout_seconds", 30),
                ssl=self.config.get("ssl_mode", None),
            )
        return self._pool

    def _build_sql(self, query: dict[str, Any]) -> str:
        if "sql" in query:
            return query["sql"]
        schema = query.get("schema", "public")
        table = query["table"]
        return f'SELECT * FROM "{schema}"."{table}"'

    def _records_to_df(self, rows: list[asyncpg.Record]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame()
        data = {k: [r[k] for r in rows] for k in rows[0].keys()}
        return pl.DataFrame(data)

    async def read(self, query: dict[str, Any], options: dict[str, Any] | None = None) -> pl.DataFrame:
        sql = self._build_sql(query)
        pool = await self._get_pool()

        async def _fetch():
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql)
                return self._records_to_df(rows)

        return await _with_retry(_fetch)

    async def read_stream(
        self, query: dict[str, Any], options: dict[str, Any] | None = None
    ) -> AsyncGenerator[pl.DataFrame, None]:
        opts = options or {}
        batch_size = opts.get("batch_size", 10_000)
        sql = self._build_sql(query)
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            async with conn.transaction():
                cursor = await conn.cursor(sql)
                while True:
                    rows = await cursor.fetch(batch_size)
                    if not rows:
                        break
                    yield self._records_to_df(rows)

    async def schema(self) -> SchemaInfo:
        query_table = self.config.get("default_table")
        query_schema = self.config.get("default_schema", "public")
        if not query_table:
            return SchemaInfo(columns=[])

        sql = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position
        """
        count_sql = f'SELECT COUNT(*) FROM "{query_schema}"."{query_table}"'
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, query_schema, query_table)
            row_count = await conn.fetchval(count_sql)

        columns = [
            ColumnMeta(
                name=r["column_name"],
                dtype=_PG_TO_POLARS.get(r["data_type"], "Utf8"),
                nullable=r["is_nullable"] == "YES",
            )
            for r in rows
        ]
        return SchemaInfo(columns=columns, row_count_estimate=row_count)

    async def test_connection(self) -> ConnectionStatus:
        t0 = time.perf_counter()
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return ConnectionStatus(ok=True, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return ConnectionStatus(ok=False, error=str(exc))

    async def write(
        self, df: pl.DataFrame, target: dict[str, Any], options: dict[str, Any] | None = None
    ) -> WriteResult:
        opts = options or {}
        table = target["table"]
        schema = target.get("schema", "public")
        write_mode = opts.get("write_mode", "append")
        batch_size = opts.get("batch_size", 5_000)
        t0 = time.perf_counter()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if write_mode == "replace":
                await conn.execute(f'TRUNCATE TABLE "{schema}"."{table}"')

            cols = df.columns
            placeholders = ", ".join(f"${i+1}" for i in range(len(cols)))
            col_names = ", ".join(f'"{c}"' for c in cols)
            insert_sql = f'INSERT INTO "{schema}"."{table}" ({col_names}) VALUES ({placeholders})'

            rows_written = 0
            for offset in range(0, len(df), batch_size):
                chunk = df.slice(offset, batch_size)
                records = [tuple(row) for row in chunk.iter_rows()]
                await conn.executemany(insert_sql, records)
                rows_written += len(records)

        return WriteResult(rows_written=rows_written, duration_s=time.perf_counter() - t0)

    async def close(self) -> None:
        if self._pool and not self._pool._closed:
            await self._pool.close()