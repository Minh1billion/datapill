import polars as pl

def estimate_batch_size(
    total_bytes: int,
    total_rows: int,
    target_mb: int = 50,
    min_batch: int = 10_000,
    max_batch: int = 500_000,
) -> int:
    if total_rows <= 0 or total_bytes <= 0:
        return min_batch
    rows = int(target_mb * 1024 * 1024 / (total_bytes / total_rows))
    return max(min_batch, min(rows, max_batch))


def records_to_df(rows) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    if hasattr(rows[0], "keys"):
        keys = list(rows[0].keys())
        columns = list(zip(*[r.values() if hasattr(r, "values") else r for r in rows]))
    else:
        return pl.DataFrame(rows)
    return pl.DataFrame(dict(zip(keys, columns)))


def estimate_from_sample(sample, batch_size: int | None) -> int:
    if batch_size is not None:
        return batch_size
    df = records_to_df(sample)
    return estimate_batch_size(
        total_bytes=df.estimated_size() * 1000,
        total_rows=len(sample),
    )