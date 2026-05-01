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