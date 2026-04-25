import asyncio
import asyncmy
import asyncmy.cursors
import random
import time
from typing import AsyncGenerator, Any

import polars as pl

from .base import BaseConnector, ColumnMeta, ConnectionStatus, SchemaInfo, WriteResult


_MYSQL_TO_POLARS: dict[str, str] = {
    "tinyint": "Int8",
    "smallint": "Int16",
    "mediumint": "Int32",
    "int": "Int32",
    "bigint": "Int64",
    "float": "Float32",
    "double": "Float64",
    "decimal": "Float64",
    "numeric": "Float64",
    "boolean": "Boolean",
    "bool": "Boolean",
    "bit": "Boolean",
    "char": "Utf8",
    "varchar": "Utf8",
    "tinytext": "Utf8",
    "text": "Utf8",
    "mediumtext": "Utf8",
    "longtext": "Utf8",
    "enum": "Utf8",
    "set": "Utf8",
    "json": "Utf8",
    "date": "Date",
    "datetime": "Datetime",
    "timestamp": "Datetime",
    "tinyblob": "Binary",
    "blob": "Binary",
    "mediumblob": "Binary",
    "longblob": "Binary",
    "binary": "Binary",
    "varbinary": "Binary",
}

_MAX_RETRIES = 3
_BASE_BACKOFF = 0.5

_RETRYABLE = (
    asyncmy.errors.OperationalError,
    asyncmy.errors.InternalError,
    ConnectionError,
    OSError,
)


async def _with_retry(coro_factory, max_retries: int = _MAX_RETRIES):
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except _RETRYABLE as exc:
            last_exc = exc
            wait = _BASE_BACKOFF * (2**attempt) + random.uniform(0, 0.3)
            await asyncio.sleep(wait)
    raise last_exc


class MySQLConnector(BaseConnector):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._pool: asyncmy.Pool | None = None

    async def _get_pool(self) -> asyncmy.Pool:
        if self._pool is None or self._pool._closed:
            self._pool = await asyncmy.create_pool(
                host=self.config["host"],
                port=self.config.get("port", 3306),
                db=self.config["database"],
                user=self.config["user"],
                password=self.config["password"],
                minsize=1,
                maxsize=self.config.get("max_connections", 5),
                connect_timeout=self.config.get("timeout_seconds", 30),
                charset="utf8mb4",
                autocommit=True,
            )
        return self._pool

    def _build_sql(self, query: dict[str, Any]) -> str:
        if "sql" in query:
            return query["sql"]
        schema = query.get("schema", self.config.get("database", ""))
        table = query["table"]
        if schema:
            return f"SELECT * FROM `{schema}`.`{table}`"
        return f"SELECT * FROM `{table}`"

    @staticmethod
    def _rows_to_df(rows: list[dict]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame()
        data = {k: [r[k] for r in rows] for k in rows[0]}
        return pl.DataFrame(data)

    async def read(
        self, query: dict[str, Any], options: dict[str, Any] | None = None
    ) -> pl.DataFrame:
        sql = self._build_sql(query)
        pool = await self._get_pool()

        async def _fetch():
            async with pool.acquire() as conn:
                async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
                    await cur.execute(sql)
                    rows = await cur.fetchall()
                    return self._rows_to_df(rows)

        return await _with_retry(_fetch)

    async def read_stream(
        self, query: dict[str, Any], options: dict[str, Any] | None = None
    ) -> AsyncGenerator[pl.DataFrame, None]:
        opts = options or {}
        batch_size = opts.get("batch_size", 10_000)
        sql = self._build_sql(query)
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
                await cur.execute(sql)
                while True:
                    rows = await cur.fetchmany(batch_size)
                    if not rows:
                        break
                    yield self._rows_to_df(rows)

    async def schema(self) -> SchemaInfo:
        table = self.config.get("default_table")
        database = self.config.get("default_schema") or self.config.get("database")
        if not table:
            return SchemaInfo(columns=[])

        col_sql = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """
        count_sql = f"SELECT COUNT(*) AS n FROM `{database}`.`{table}`"
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
                await cur.execute(col_sql, (database, table))
                col_rows = await cur.fetchall()
                await cur.execute(count_sql)
                count_row = await cur.fetchone()

        columns = [
            ColumnMeta(
                name=r.get("column_name") or r.get("COLUMN_NAME"),
                dtype=_MYSQL_TO_POLARS.get(
                    (r.get("data_type") or r.get("DATA_TYPE", "")).lower(), "Utf8"
                ),
                nullable=(r.get("is_nullable") or r.get("IS_NULLABLE", "YES")) == "YES",
            )
            for r in col_rows
        ]
        row_count = count_row["n"] if count_row else None
        return SchemaInfo(columns=columns, row_count_estimate=row_count)

    async def test_connection(self) -> ConnectionStatus:
        t0 = time.perf_counter()
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
                    await cur.execute("SELECT 1")
            return ConnectionStatus(ok=True, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return ConnectionStatus(ok=False, error=str(exc))

    async def write(
        self,
        df: pl.DataFrame,
        target: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> WriteResult:
        opts = options or {}
        table = target["table"]
        schema = target.get("schema", self.config.get("database", ""))
        write_mode = opts.get("write_mode", "append")
        batch_size = opts.get("batch_size", 5_000)
        qualified = f"`{schema}`.`{table}`" if schema else f"`{table}`"

        t0 = time.perf_counter()
        pool = await self._get_pool()

        cols = df.columns
        col_names = ", ".join(f"`{c}`" for c in cols)
        placeholders = ", ".join(["%s"] * len(cols))
        insert_sql = f"INSERT INTO {qualified} ({col_names}) VALUES ({placeholders})"

        async with pool.acquire() as conn:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
                if write_mode == "replace":
                    await cur.execute(f"TRUNCATE TABLE {qualified}")

                rows_written = 0
                for offset in range(0, len(df), batch_size):
                    chunk = df.slice(offset, batch_size)
                    records = [tuple(row) for row in chunk.iter_rows()]
                    await cur.executemany(insert_sql, records)
                    rows_written += len(records)

            await conn.commit()

        return WriteResult(rows_written=rows_written, duration_s=time.perf_counter() - t0)

    async def close(self) -> None:
        if self._pool and not self._pool._closed:
            self._pool.close()
            await self._pool.wait_closed()