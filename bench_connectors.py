"""
bench_connectors.py — đặt ở root project, chạy:
    python bench_connectors.py

Yêu cầu docker-compose.test.yml đang healthy.
RestAPI dùng mock aiohttp server nội bộ, không cần service ngoài.

Đo tốc độ (rows/s) và RAM delta (MB) cho mỗi connector ở cả 2 mode:
    stream=False  →  1 DataFrame (toàn bộ dữ liệu, 1 lần)
    stream=True   →  AsyncGenerator[DataFrame, ...]  (chunk theo chunk)

Tradeoff rõ ràng:
    stream=False  →  tốc độ cao hơn, RAM spike lớn hơn
    stream=True   →  RAM ổn định hơn, overhead nhiều round-trip / chunk hơn

1,000,000 rows · 5 columns (user_id, category, value, score, label)
"""

from __future__ import annotations

import asyncio
import gc
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Any

import polars as pl
import psutil
from aiohttp import web

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datapill.connectors.postgresql import PostgreSqlConnector, PostgreSQLConnectorConfig
from datapill.connectors.mysql import MySQLConnector, MySQLConnectorConfig
from datapill.connectors.sqlite import SQLiteConnector, SQLiteConnectorConfig
from datapill.connectors.s3 import S3Connector, S3ConnectorConfig
from datapill.connectors.kafka import KafkaConnector, KafkaConnectorConfig
from datapill.connectors.rest_api import RestApiConnector, RestApiConnectorConfig
from datapill.connectors.local_directory import LocalDirectoryConnector, LocalConnectorConfig


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

TOTAL_ROWS   = 1_000_000
INSERT_BATCH = 50_000
TABLE        = "bench_data"
KAFKA_TOPIC  = "bench_topic"
REST_PORT    = 18888
REST_PAGE    = 10_000

PG_CFG = PostgreSQLConnectorConfig(
    host="localhost", port=5433,
    database="testdb", user="testuser", password="testpass",
    fetch_size=50_000,
)
MY_CFG = MySQLConnectorConfig(
    host="localhost", port=3307,
    database="testdb", user="testuser", password="testpass",
    fetch_size=50_000,
)
SQLITE_PATH = Path(tempfile.gettempdir()) / "bench_connectors.db"
SQLITE_CFG  = SQLiteConnectorConfig(path=str(SQLITE_PATH), fetch_size=50_000)

S3_CFG = S3ConnectorConfig(
    bucket="testbucket",
    region="us-east-1",
    access_key="minioadmin",
    secret_key="minioadmin",
    endpoint_url="http://localhost:9000",
)
KAFKA_CFG = KafkaConnectorConfig(
    brokers=["localhost:9092"],
    group_id="bench_group",
    auto_offset_reset="earliest",
    max_poll_records=5_000,
)

LOCAL_DIR = Path(tempfile.gettempdir()) / "bench_local"
LOCAL_CFG = LocalConnectorConfig(base_path=str(LOCAL_DIR), mkdir=True)

REST_CFG = RestApiConnectorConfig(
    base_url=f"http://localhost:{REST_PORT}",
    pagination_type="page",
    page_param="page",
    page_size_param="page_size",
    page_size=REST_PAGE,
    results_key="data",
)

SQL_SELECT      = f"SELECT * FROM {TABLE}"
SQL_SELECT_LITE = f"SELECT * FROM _src.{TABLE}"

CREATE_PG = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id         SERIAL PRIMARY KEY,
        user_id    INTEGER NOT NULL,
        category   VARCHAR(32) NOT NULL,
        value      DOUBLE PRECISION NOT NULL,
        score      DOUBLE PRECISION NOT NULL,
        label      VARCHAR(32) NOT NULL
    )
"""
CREATE_MY = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        user_id    INT NOT NULL,
        category   VARCHAR(32) NOT NULL,
        value      DOUBLE NOT NULL,
        score      DOUBLE NOT NULL,
        label      VARCHAR(32) NOT NULL
    )
"""
CREATE_LITE = f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL,
        category   TEXT NOT NULL,
        value      REAL NOT NULL,
        score      REAL NOT NULL,
        label      TEXT NOT NULL
    )
"""
INSERT_PG   = f"INSERT INTO {TABLE}(user_id,category,value,score,label) VALUES($1,$2,$3,$4,$5)"
INSERT_MY   = f"INSERT INTO {TABLE}(user_id,category,value,score,label) VALUES(%s,%s,%s,%s,%s)"
INSERT_LITE = f"INSERT INTO {TABLE}(user_id,category,value,score,label) VALUES(?,?,?,?,?)"


# ══════════════════════════════════════════════════════════════════════════════
# Measurement
# ══════════════════════════════════════════════════════════════════════════════

def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


@dataclass
class BenchResult:
    label: str
    rows: int
    wall_s: float
    ram_delta_mb: float
    error: str | None = None

    @property
    def rows_per_sec(self) -> float:
        return self.rows / self.wall_s if self.wall_s else 0.0


async def _wrap_sync_gen(gen) -> AsyncGenerator[pl.DataFrame, Any]:
    for chunk in gen:
        yield chunk


async def _drain(result, is_stream: bool) -> int:
    if not is_stream:
        return len(result)
    if hasattr(result, "__aiter__"):
        total = 0
        async for chunk in result:
            total += len(chunk)
        return total
    total = 0
    async for chunk in _wrap_sync_gen(result):
        total += len(chunk)
    return total


async def _measure(label: str, coro, is_stream: bool) -> BenchResult:
    gc.collect()
    ram0 = _rss_mb()
    t0   = time.perf_counter()
    rows = 0
    err  = None
    try:
        result = await coro
        rows = await _drain(result, is_stream)
    except Exception as e:
        err = str(e)
    wall = time.perf_counter() - t0
    gc.collect()
    return BenchResult(
        label=label, rows=rows, wall_s=wall,
        ram_delta_mb=_rss_mb() - ram0, error=err,
    )


def _print_result(r: BenchResult):
    if r.error:
        print(f"  FAIL  {r.label}")
        print(f"        {r.error[:120]}")
        return
    print(
        f"  {r.rows_per_sec:>13,.0f} rows/s  "
        f"wall={r.wall_s:.3f}s  "
        f"RAM{r.ram_delta_mb:+.1f}MB  "
        f"│  {r.label}"
    )


def _print_summary(results: list[BenchResult]):
    valid = [r for r in results if not r.error and r.rows > 0]
    valid.sort(key=lambda r: r.wall_s)
    sep = "═" * 92

    print(f"\n\n{sep}")
    print(f" SUMMARY — {TOTAL_ROWS:,} rows")
    print(sep)
    print(f"{'Connector / Mode':<48} {'Rows/s':>13} {'Wall(s)':>9} {'RAM Δ MB':>10}")
    print("─" * 92)
    for r in valid:
        print(
            f"{r.label:<48} {r.rows_per_sec:>13,.0f} "
            f"{r.wall_s:>9.3f} {r.ram_delta_mb:>+10.1f}"
        )
    print(sep)
    if valid:
        fastest   = valid[0]
        most_mem  = max(valid, key=lambda r: r.ram_delta_mb)
        least_mem = min(valid, key=lambda r: r.ram_delta_mb)
        print(f"  Fastest   : {fastest.label} ({fastest.wall_s:.3f}s)")
        print(f"  Most RAM  : {most_mem.label} ({most_mem.ram_delta_mb:+.1f} MB)")
        print(f"  Least RAM : {least_mem.label} ({least_mem.ram_delta_mb:+.1f} MB)")

    failed = [r for r in results if r.error]
    if failed:
        print("\n  FAILED:")
        for r in failed:
            print(f"    {r.label}: {r.error[:100]}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _gen_rows(offset: int, n: int) -> list[tuple]:
    return [
        (
            (offset + i) % 10_000,
            f"cat_{(offset + i) % 20}",
            float((offset + i) % 1_000) / 10.0,
            float((offset + i) % 500) / 5.0,
            f"label_{(offset + i) % 50}",
        )
        for i in range(n)
    ]


def _gen_df(n: int = TOTAL_ROWS) -> pl.DataFrame:
    rows = _gen_rows(0, n)
    return pl.DataFrame({
        "user_id":  [r[0] for r in rows],
        "category": [r[1] for r in rows],
        "value":    [r[2] for r in rows],
        "score":    [r[3] for r in rows],
        "label":    [r[4] for r in rows],
    })


# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════

async def _bulk_insert(connector, sql: str, inserted_start: int = 0):
    inserted = inserted_start
    while inserted < TOTAL_ROWS:
        n = min(INSERT_BATCH, TOTAL_ROWS - inserted)
        await connector.execute(sql, params=_gen_rows(inserted, n), many=True)
        inserted += n
        if inserted % 200_000 == 0:
            print(f"    {inserted:>10,} / {TOTAL_ROWS:,}")


async def setup_postgres(c: PostgreSqlConnector):
    print("  drop + create + insert 1M rows...")
    await c.execute(f"DROP TABLE IF EXISTS {TABLE}")
    await c.execute(CREATE_PG)
    await _bulk_insert(c, INSERT_PG)
    print("  done.\n")


async def setup_mysql(c: MySQLConnector):
    print("  drop + create + insert 1M rows...")
    await c.execute(f"DROP TABLE IF EXISTS {TABLE}")
    await c.execute(CREATE_MY)
    await _bulk_insert(c, INSERT_MY)
    print("  done.\n")


def setup_sqlite():
    print("  creating DB + insert 1M rows...")
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(CREATE_LITE)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_user ON {TABLE}(user_id)")
    inserted = 0
    while inserted < TOTAL_ROWS:
        n = min(INSERT_BATCH, TOTAL_ROWS - inserted)
        conn.executemany(INSERT_LITE, _gen_rows(inserted, n))
        conn.commit()
        inserted += n
        if inserted % 200_000 == 0:
            print(f"    {inserted:>10,} / {TOTAL_ROWS:,}")
    conn.close()
    print("  done.\n")


async def setup_s3(c: S3Connector):
    print("  uploading parquet + csv (1M rows)...")
    df = _gen_df()
    await c.write(df, "bench_1m.parquet", format="parquet")
    await c.write(df, "bench_1m.csv",     format="csv")
    del df; gc.collect()
    print("  done.\n")


def setup_local():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print("  writing parquet + csv (1M rows)...")
    df = _gen_df()
    df.write_parquet(LOCAL_DIR / "bench_1m.parquet")
    df.write_csv(LOCAL_DIR / "bench_1m.csv")
    del df; gc.collect()
    print("  done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# RestAPI mock server
# ══════════════════════════════════════════════════════════════════════════════

async def _start_rest_server() -> web.AppRunner:
    app = web.Application()

    async def handle_items(req: web.Request) -> web.Response:
        page      = int(req.rel_url.query.get("page", 1))
        page_size = int(req.rel_url.query.get("page_size", REST_PAGE))
        offset    = (page - 1) * page_size
        if offset >= TOTAL_ROWS:
            return web.json_response({"data": [], "total": TOTAL_ROWS})
        n    = min(page_size, TOTAL_ROWS - offset)
        rows = _gen_rows(offset, n)
        data = [
            {"user_id": r[0], "category": r[1],
             "value": r[2], "score": r[3], "label": r[4]}
            for r in rows
        ]
        return web.json_response({"data": data, "total": TOTAL_ROWS})

    async def handle_root(req: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app.router.add_get("/items", handle_items)
    app.router.add_get("/",      handle_root)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "localhost", REST_PORT).start()
    return runner


# ══════════════════════════════════════════════════════════════════════════════
# REST fetch-all helper (stream=False phải fetch đủ 1M rows để so sánh công bằng)
# ══════════════════════════════════════════════════════════════════════════════

async def _rest_fetch_all(rest: RestApiConnector) -> pl.DataFrame:
    frames = []
    page = 1
    while True:
        chunk = await rest.query(
            "items",
            params={"page": page, "page_size": REST_PAGE},
            stream=False,
        )
        if len(chunk) == 0:
            break
        frames.append(chunk)
        page += 1
    return pl.concat(frames) if frames else pl.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    results: list[BenchResult] = []

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    print("═" * 64)
    print("POSTGRESQL")
    print("═" * 64)
    pg = PostgreSqlConnector(PG_CFG)
    s  = await pg.connect()
    if not s.ok:
        print(f"  SKIP — {s.error}\n")
    else:
        print(f"  connected ({s.latency_ms:.1f}ms)")
        await setup_postgres(pg)
        for label, stream in [
            ("postgresql  stream=False (COPY CSV)",      False),
            ("postgresql  stream=True  (server cursor)", True),
        ]:
            r = await _measure(label, pg.query(SQL_SELECT, stream=stream), stream)
            _print_result(r); results.append(r)
        await pg.cleanup()

    # ── MySQL ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("MYSQL")
    print("═" * 64)
    my = MySQLConnector(MY_CFG)
    s  = await my.connect()
    if not s.ok:
        print(f"  SKIP — {s.error}\n")
    else:
        print(f"  connected ({s.latency_ms:.1f}ms)")
        await setup_mysql(my)
        for label, stream in [
            ("mysql       stream=False (connectorx)", False),
            ("mysql       stream=True  (SSCursor)",   True),
        ]:
            r = await _measure(label, my.query(SQL_SELECT, stream=stream), stream)
            _print_result(r); results.append(r)
        await my.cleanup()

    # ── SQLite ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("SQLITE")
    print("═" * 64)
    setup_sqlite()
    lite = SQLiteConnector(SQLITE_CFG)
    s    = await lite.connect()
    if not s.ok:
        print(f"  SKIP — {s.error}\n")
    else:
        print(f"  connected ({s.latency_ms:.1f}ms)")
        for label, stream in [
            ("sqlite      stream=False (duckdb ATTACH scan)",  False),
            ("sqlite      stream=True  (duckdb arrow reader)", True),
        ]:
            r = await _measure(label, lite.query(SQL_SELECT_LITE, stream=stream), stream)
            _print_result(r); results.append(r)
        await lite.cleanup()
        if SQLITE_PATH.exists():
            SQLITE_PATH.unlink()

    # ── S3 / MinIO ──────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("S3 / MINIO")
    print("═" * 64)
    s3 = S3Connector(S3_CFG)
    s  = await s3.connect()
    if not s.ok:
        print(f"  SKIP — {s.error}\n")
    else:
        print(f"  connected ({s.latency_ms:.1f}ms)")
        await setup_s3(s3)
        for label, path, fmt, stream in [
            ("s3          stream=False parquet", "bench_1m.parquet", "parquet", False),
            ("s3          stream=True  parquet", "bench_1m.parquet", "parquet", True),
            ("s3          stream=False csv",     "bench_1m.csv",     "csv",     False),
            ("s3          stream=True  csv",     "bench_1m.csv",     "csv",     True),
        ]:
            r = await _measure(label, s3.read(path, format=fmt, stream=stream), stream)
            _print_result(r); results.append(r)
        await s3.cleanup()

    # ── Kafka ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("KAFKA")
    print("═" * 64)
    kaf = KafkaConnector(KAFKA_CFG)
    s   = await kaf.connect()
    if not s.ok:
        print(f"  SKIP — {s.error}\n")
    else:
        print(f"  connected ({s.latency_ms:.1f}ms)")

        df_kaf = _gen_df()
        gc.collect(); ram0 = _rss_mb(); t0 = time.perf_counter()
        err = None; sent = 0
        try:
            sent = await kaf.produce(KAFKA_TOPIC, df_kaf)
        except Exception as e:
            err = str(e)
        wall = time.perf_counter() - t0
        r = BenchResult("kafka       produce", sent, wall, _rss_mb() - ram0, err)
        _print_result(r); results.append(r)
        del df_kaf; gc.collect()

        gc.collect(); ram0 = _rss_mb(); t0 = time.perf_counter()
        consumed = 0; err = None
        try:
            async for chunk in kaf.consume(KAFKA_TOPIC, max_messages=TOTAL_ROWS, timeout_ms=8_000):
                consumed += len(chunk)
        except Exception as e:
            err = str(e)
        wall = time.perf_counter() - t0
        r = BenchResult("kafka       consume stream=True", consumed, wall, _rss_mb() - ram0, err)
        _print_result(r); results.append(r)

        await kaf.cleanup()

    # ── RestAPI (mock server) ───────────────────────────────────────────────
    print("\n" + "═" * 64)
    print(f"REST API  (mock localhost:{REST_PORT})")
    print("═" * 64)
    rest_runner = await _start_rest_server()
    await asyncio.sleep(0.1)

    rest = RestApiConnector(REST_CFG)
    s    = await rest.connect()
    if not s.ok:
        print(f"  SKIP — {s.error}\n")
        await rest_runner.cleanup()
    else:
        print(f"  connected ({s.latency_ms:.1f}ms)")
        print(f"  {TOTAL_ROWS // REST_PAGE} pages × {REST_PAGE:,} rows = {TOTAL_ROWS:,} total")

        gc.collect(); ram0 = _rss_mb(); t0 = time.perf_counter()
        err = None; total_rows = 0
        try:
            df_all = await _rest_fetch_all(rest)
            total_rows = len(df_all)
            del df_all
        except Exception as e:
            err = str(e)
        wall = time.perf_counter() - t0
        r = BenchResult(
            f"rest_api    stream=False (all {TOTAL_ROWS // REST_PAGE} pages concat)",
            total_rows, wall, _rss_mb() - ram0, err,
        )
        _print_result(r); results.append(r)

        r = await _measure(
            "rest_api    stream=True  (paginate all)",
            rest.query("items", stream=True),
            is_stream=True,
        )
        _print_result(r); results.append(r)

        await rest_runner.cleanup()
        await rest.cleanup()

    # ── Local Directory ─────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("LOCAL DIRECTORY")
    print("═" * 64)
    setup_local()
    loc = LocalDirectoryConnector(LOCAL_CFG)
    s   = await loc.connect()
    if not s.ok:
        print(f"  SKIP — {s.error}\n")
    else:
        print(f"  connected ({s.latency_ms:.1f}ms)")
        for label, path, fmt, stream in [
            ("local       stream=False parquet", "bench_1m.parquet", "parquet", False),
            ("local       stream=True  parquet", "bench_1m.parquet", "parquet", True),
            ("local       stream=False csv",     "bench_1m.csv",     "csv",     False),
            ("local       stream=True  csv",     "bench_1m.csv",     "csv",     True),
        ]:
            r = await _measure(label, loc.read(path, format=fmt, stream=stream), stream)
            _print_result(r); results.append(r)
        await loc.cleanup()
        for f in LOCAL_DIR.iterdir():
            f.unlink()
        LOCAL_DIR.rmdir()

    _print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())