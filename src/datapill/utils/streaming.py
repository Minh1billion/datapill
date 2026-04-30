import polars as pl
import asyncpg


def estimate_batch_size(
    total_bytes: int,
    total_rows: int,
    target_mb: int = 50,
    min_batch: int = 10_000,
    max_batch: int = 500_000,
) -> int:
    if total_rows <= 0 or total_bytes <= 0:
        return min_batch
    target_bytes = target_mb * 1024 * 1024
    bytes_per_row = total_bytes / total_rows
    rows = int(target_bytes / bytes_per_row)
    return max(min_batch, min(rows, max_batch))


def records_to_df(rows):
    if not rows:
        return pl.DataFrame()
    keys = list(rows[0].keys())
    columns = list(zip(*rows))
    return pl.DataFrame(dict(zip(keys, columns)))


def estimate_from_sample(
    sample: list[asyncpg.Record],
    batch_size: int | None,
) -> int:
    if batch_size is not None:
        return batch_size
    keys = list(sample[0].keys())
    sample_df = pl.DataFrame({k: [r[k] for r in sample] for k in keys})
    return estimate_batch_size(
        total_bytes=sample_df.estimated_size() * 1000,
        total_rows=len(sample),
    )