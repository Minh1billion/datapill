"""
bench_s3_stream.py — thử các cách read parquet từ S3/MinIO stream=True
Chạy: python bench_s3_stream.py

Yêu cầu MinIO đang chạy tại localhost:9000
"""

import asyncio
import gc
import io
import os
import time

import aioboto3
import polars as pl
import psutil
import pyarrow.dataset as pad
import pyarrow.parquet as pq
import s3fs

BUCKET       = "testbucket"
KEY          = "bench_stream_1m.parquet"
ENDPOINT     = "http://localhost:9000"
ACCESS_KEY   = "minioadmin"
SECRET_KEY   = "minioadmin"
REGION       = "us-east-1"
TOTAL_ROWS   = 1_000_000


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def gen_df() -> pl.DataFrame:
    n = TOTAL_ROWS
    idx = range(n)
    return pl.DataFrame({
        "user_id":  [i % 10_000 for i in idx],
        "category": [f"cat_{i % 20}" for i in idx],
        "value":    [float(i % 1_000) / 10.0 for i in idx],
        "score":    [float(i % 500) / 5.0 for i in idx],
        "label":    [f"label_{i % 50}" for i in idx],
    })


def boto_session():
    return aioboto3.Session(
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
    )


async def upload():
    print("uploading bench_stream_1m.parquet ...")
    df = gen_df()
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    async with boto_session().client("s3", endpoint_url=ENDPOINT) as s3:
        try:
            await s3.head_bucket(Bucket=BUCKET)
        except Exception:
            await s3.create_bucket(Bucket=BUCKET)
        await s3.put_object(Bucket=BUCKET, Key=KEY, Body=buf.read())
    print("done.\n")


# ── các cách read ────────────────────────────────────────────────────────────

async def way_download_then_pq_iter_batches() -> int:
    async with boto_session().client("s3", endpoint_url=ENDPOINT) as s3:
        obj = await s3.get_object(Bucket=BUCKET, Key=KEY)
        data = await obj["Body"].read()
    total = 0
    for batch in pq.ParquetFile(io.BytesIO(data)).iter_batches():
        total += len(pl.from_arrow(batch))
    return total


async def way_download_then_iter_slices() -> int:
    async with boto_session().client("s3", endpoint_url=ENDPOINT) as s3:
        obj = await s3.get_object(Bucket=BUCKET, Key=KEY)
        data = await obj["Body"].read()
    df = pl.read_parquet(io.BytesIO(data))
    total = 0
    for chunk in df.iter_slices(50_000):
        total += len(chunk)
    return total


async def way_s3fs_dataset_to_batches() -> int:
    fs = s3fs.S3FileSystem(
        key=ACCESS_KEY, secret=SECRET_KEY,
        endpoint_url=ENDPOINT,
        client_kwargs={"region_name": REGION},
    )
    dataset = pad.dataset(f"{BUCKET}/{KEY}", filesystem=fs, format="parquet")
    total = 0
    for batch in dataset.to_batches():
        total += len(pl.from_arrow(batch))
    return total


async def way_s3fs_open_pq_iter_batches() -> int:
    fs = s3fs.S3FileSystem(
        key=ACCESS_KEY, secret=SECRET_KEY,
        endpoint_url=ENDPOINT,
        client_kwargs={"region_name": REGION},
    )
    with fs.open(f"{BUCKET}/{KEY}", "rb") as f:
        pf = pq.ParquetFile(f)
        total = 0
        for batch in pf.iter_batches():
            total += len(pl.from_arrow(batch))
    return total


async def way_s3fs_open_pq_iter_batches_to_thread() -> int:
    import asyncio
    fs = s3fs.S3FileSystem(
        key=ACCESS_KEY, secret=SECRET_KEY,
        endpoint_url=ENDPOINT,
        client_kwargs={"region_name": REGION},
    )
    def read_sync():
        with fs.open(f"{BUCKET}/{KEY}", "rb") as f:
            pf = pq.ParquetFile(f)
            frames = [pl.from_arrow(b) for b in pf.iter_batches()]
        return sum(len(fr) for fr in frames)
    return await asyncio.to_thread(read_sync)


async def way_aioboto3_multipart_stream() -> int:
    async with boto_session().client("s3", endpoint_url=ENDPOINT) as s3:
        obj = await s3.get_object(Bucket=BUCKET, Key=KEY)
        chunks = []
        async for chunk in obj["Body"].iter_chunks(1024 * 1024):
            chunks.append(chunk)
    data = b"".join(chunks)
    df = pl.read_parquet(io.BytesIO(data))
    total = 0
    for chunk in df.iter_slices(50_000):
        total += len(chunk)
    return total


async def way_polars_scan_parquet_s3() -> int:
    url = f"s3://{BUCKET}/{KEY}"
    df = pl.scan_parquet(
        url,
        storage_options={
            "aws_access_key_id": ACCESS_KEY,
            "aws_secret_access_key": SECRET_KEY,
            "endpoint_url": ENDPOINT,
            "region_name": REGION,
        },
    )
    total = 0
    for chunk in df.collect().iter_slices(50_000):
        total += len(chunk)
    return total


# ── runner ───────────────────────────────────────────────────────────────────

WAYS = [
    ("download → pq.iter_batches",          way_download_then_pq_iter_batches),
    ("download → iter_slices(50k)",          way_download_then_iter_slices),
    ("s3fs dataset.to_batches",              way_s3fs_dataset_to_batches),
    ("s3fs open → pq.iter_batches",          way_s3fs_open_pq_iter_batches),
    ("s3fs open → pq.iter_batches to_thread",way_s3fs_open_pq_iter_batches_to_thread),
    ("aioboto3 iter_chunks → iter_slices",   way_aioboto3_multipart_stream),
    ("polars scan_parquet s3://",            way_polars_scan_parquet_s3),
]


async def main():
    await upload()

    sep = "═" * 80
    print(sep)
    print(f"  {'Method':<44} {'Rows/s':>12} {'Wall(s)':>8} {'RAM Δ MB':>9}")
    print("─" * 80)

    for label, fn in WAYS:
        gc.collect()
        ram0 = rss_mb()
        t0   = time.perf_counter()
        rows = 0
        err  = None
        try:
            rows = await fn()
        except Exception as e:
            err = str(e)
        wall = time.perf_counter() - t0
        gc.collect()
        ram_delta = rss_mb() - ram0

        if err:
            print(f"  {'FAIL: ' + label:<44} {err[:30]}")
        else:
            rps = rows / wall if wall else 0
            print(f"  {label:<44} {rps:>12,.0f} {wall:>8.3f} {ram_delta:>+9.1f}")

    print(sep)


if __name__ == "__main__":
    asyncio.run(main())