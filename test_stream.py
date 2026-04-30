import asyncio, time
from typing import Optional, AsyncGenerator, Any
import polars as pl
import asyncpg

DSN = "postgresql://testuser:testpass@localhost:5433/testdb"

def estimate_batch_size(total_bytes, total_rows, target_mb=50,
                        min_batch=10_000, max_batch=500_000):
    if total_rows <= 0 or total_bytes <= 0:
        return min_batch
    rows = int(target_mb * 1024 * 1024 / (total_bytes / total_rows))
    return max(min_batch, min(rows, max_batch))

def records_to_df(rows):
    if not rows:
        return pl.DataFrame()
    keys = list(rows[0].keys())
    return pl.DataFrame({k: [r[k] for r in rows] for k in keys})

def estimate_from_sample(sample, batch_size):
    if batch_size is not None:
        return batch_size
    keys = list(sample[0].keys())
    df = pl.DataFrame({k: [r[k] for r in sample] for k in keys})
    return estimate_batch_size(df.estimated_size() * 1000, len(sample))

async def query(pool, sql, params=None, stream=False, batch_size=None):
    if not stream:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params or [])
        return records_to_df(rows)

    async def _stream():
        async with pool.acquire() as conn:
            async with conn.transaction():
                cursor = await conn.cursor(sql, *params or [])
                sample = await cursor.fetch(1000)
                if not sample:
                    return
                resolved = estimate_from_sample(sample, batch_size)
                print(f"  [batch_size = {resolved:,}]")
                yield records_to_df(sample)
                while True:
                    rows = await cursor.fetch(resolved)
                    if not rows:
                        break
                    yield records_to_df(rows)
    return _stream()

async def setup(pool):
    print("Inserting 1,000,000 rows...")
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS bench")
        await conn.execute("""
            CREATE TABLE bench (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                score DOUBLE PRECISION NOT NULL,
                category TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)
        t0 = time.perf_counter()
        await conn.execute("""
            INSERT INTO bench (name, score, category)
            SELECT 'user_' || i, random()*1000,
                   CASE (i%5) WHEN 0 THEN 'alpha' WHEN 1 THEN 'beta'
                               WHEN 2 THEN 'gamma' WHEN 3 THEN 'delta'
                               ELSE 'epsilon' END
            FROM generate_series(1, 1000000) i
        """)
    print(f"  Done in {time.perf_counter()-t0:.2f}s\n")

async def bench(pool, label, stream, batch_size=None):
    print("=" * 50)
    print(f"TEST: {label}")
    t0 = time.perf_counter()
    total, batches = 0, 0
    if not stream:
        df = await query(pool, "SELECT * FROM bench")
        total, batches = len(df), 1
        print(f"  Memory : {df.estimated_size()/1024/1024:.1f} MB")
    else:
        gen = await query(pool, "SELECT * FROM bench", stream=True, batch_size=batch_size)
        async for batch in gen:
            total += len(batch)
            batches += 1
    elapsed = time.perf_counter() - t0
    print(f"  Rows   : {total:,}  |  Batches : {batches}")
    print(f"  Time   : {elapsed:.3f}s  |  Throughput : {total/elapsed:,.0f} rows/s\n")

async def main():
    pool = await asyncpg.create_pool(dsn=DSN, min_size=1, max_size=10)
    try:
        await setup(pool)
        await bench(pool, "Non-stream", stream=False)
        await bench(pool, "Stream auto batch", stream=True)
        await bench(pool, "Stream fixed 50k", stream=True, batch_size=50_000)
        await bench(pool, "Stream fixed 100k", stream=True, batch_size=100_000)
    finally:
        await pool.close()

asyncio.run(main())
