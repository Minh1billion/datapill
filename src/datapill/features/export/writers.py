from pathlib import Path
from typing import Any

import polars as pl


def write_csv(df: pl.DataFrame, path: Path, **opts: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(
        path,
        separator=opts.get("delimiter", ","),
    )


def write_parquet(df: pl.DataFrame, path: Path, **opts: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partition_by: list[str] | None = opts.get("partition_by")
    if partition_by:
        df.write_parquet(
            path,
            compression=opts.get("compression", "snappy"),
            row_group_size=opts.get("row_group_size"),
            partition_by=partition_by,
        )
    else:
        df.write_parquet(
            path,
            compression=opts.get("compression", "snappy"),
            row_group_size=opts.get("row_group_size"),
        )


def write_excel(df: pl.DataFrame, path: Path, **opts: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_excel(
        path,
        worksheet=opts.get("sheet_name", "Sheet1"),
        freeze_panes=(1, 0) if opts.get("freeze_header", True) else None,
    )


def write_json(df: pl.DataFrame, path: Path, **opts: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    use_ndjson = (
        opts.get("jsonl")
        or opts.get("orient") == "lines"
        or Path(path).suffix.lower() in (".jsonl", ".ndjson")
    )
    if use_ndjson:
        df.write_ndjson(path)
    else:
        df.write_json(path)


def write_arrow(df: pl.DataFrame, path: Path, **opts: Any) -> None:
    import pyarrow as pa
    import pyarrow.feather as feather

    path.parent.mkdir(parents=True, exist_ok=True)
    table = df.to_arrow()
    feather.write_feather(
        table,
        str(path),
        compression=opts.get("compression", "zstd"),
    )


_FORMAT_WRITERS = {
    "csv": write_csv,
    "parquet": write_parquet,
    "excel": write_excel,
    "xlsx": write_excel,
    "json": write_json,
    "jsonl": write_json,
    "arrow": write_arrow,
    "feather": write_arrow,
}


def write(df: pl.DataFrame, path: Path, fmt: str, **opts: Any) -> None:
    writer = _FORMAT_WRITERS.get(fmt.lower())
    if writer is None:
        raise ValueError(f"Unsupported format: '{fmt}'. Supported: {sorted(_FORMAT_WRITERS)}")
    writer(df, path, **opts)


def read(path: Path, fmt: str) -> pl.DataFrame:
    fmt = fmt.lower()
    if fmt == "csv":
        return pl.read_csv(path)
    if fmt == "parquet":
        return pl.read_parquet(path)
    if fmt in ("excel", "xlsx"):
        return pl.read_excel(path)
    if fmt == "json":
        return pl.read_json(path)
    if fmt == "jsonl":
        return pl.read_ndjson(path)
    if fmt in ("arrow", "feather"):
        import pyarrow.feather as feather
        return pl.from_arrow(feather.read_table(str(path)))
    raise ValueError(f"Unsupported format for read: '{fmt}'")