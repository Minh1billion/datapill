import time
from pathlib import Path
from typing import AsyncGenerator, Any
import polars as pl
from .base import BaseConnector, SchemaInfo, ColumnMeta, ConnectionStatus, WriteResult

_SUPPORTED = {".csv", ".parquet", ".xlsx", ".xls", ".json", ".ndjson"}

_ADAPTIVE_MIN_BATCH = 10_000
_ADAPTIVE_MAX_BATCH = 500_000
_ADAPTIVE_TARGET_MB  = 50 


def _adaptive_batch_size(df: pl.DataFrame, hint: int | None) -> int:
    if hint is not None:
        return max(_ADAPTIVE_MIN_BATCH, min(hint, _ADAPTIVE_MAX_BATCH))
    if len(df) == 0:
        return _ADAPTIVE_MIN_BATCH
    mem_mb = df.estimated_size("mb")
    if mem_mb <= 0:
        return _ADAPTIVE_MIN_BATCH
    bytes_per_row = (mem_mb * 1_048_576) / len(df)
    target_rows = int((_ADAPTIVE_TARGET_MB * 1_048_576) / bytes_per_row)
    return max(_ADAPTIVE_MIN_BATCH, min(target_rows, _ADAPTIVE_MAX_BATCH))


def _scan_csv(path: Path, options: dict[str, Any]) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        separator=options.get("delimiter", ","),
        has_header=options.get("has_header", True),
        n_rows=options.get("n_rows", None),
        infer_schema_length=options.get("infer_schema_length", 10_000),
        try_parse_dates=options.get("try_parse_dates", True),
        encoding=options.get("encoding", "utf8"),
        null_values=options.get("null_values", None),
        ignore_errors=options.get("ignore_errors", False),
    )


def _read_eager(path: Path, options: dict[str, Any]) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _scan_csv(path, options).collect()
    if suffix == ".parquet":
        return pl.read_parquet(path, n_rows=options.get("n_rows", None))
    if suffix in (".xlsx", ".xls"):
        return pl.read_excel(path, sheet_id=options.get("sheet_id", 1))
    if suffix in (".json", ".ndjson"):
        return pl.read_ndjson(path)
    raise ValueError(f"Unsupported file format: {suffix}")


class LocalFileConnector(BaseConnector):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    async def read(self, query: dict[str, Any], options: dict[str, Any] | None = None) -> pl.DataFrame:
        return _read_eager(Path(query["path"]), options or {})

    async def read_stream(
        self, query: dict[str, Any], options: dict[str, Any] | None = None
    ) -> AsyncGenerator[pl.DataFrame, None]:
        opts = options or {}
        path = Path(query["path"])
        hint_batch = opts.get("batch_size", None)

        df = _read_eager(path, opts)

        batch_size = _adaptive_batch_size(df, hint_batch)
        total = len(df)

        for offset in range(0, total, batch_size):
            yield df.slice(offset, batch_size)

    async def schema(self) -> SchemaInfo:
        path = Path(self.config.get("default_path", "."))
        if not path.is_file():
            return SchemaInfo(columns=[])
        suffix = path.suffix.lower()
        if suffix in (".csv", ".parquet", ".json", ".ndjson"):
            polars_schema = _scan_csv(path, {}).collect_schema() if suffix == ".csv" \
                else pl.scan_parquet(path).collect_schema()
        else:
            polars_schema = _read_eager(path, {"n_rows": 0}).schema
        return SchemaInfo(columns=[
            ColumnMeta(name=k, dtype=str(v), nullable=True)
            for k, v in polars_schema.items()
        ])

    async def test_connection(self) -> ConnectionStatus:
        t0 = time.perf_counter()
        try:
            ok = Path(self.config.get("default_path", ".")).exists()
            return ConnectionStatus(ok=ok, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return ConnectionStatus(ok=False, error=str(exc))

    async def write(
        self, df: pl.DataFrame, target: dict[str, Any], options: dict[str, Any] | None = None
    ) -> WriteResult:
        opts = options or {}
        path = Path(target["path"])
        t0 = time.perf_counter()
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df.write_csv(path, separator=opts.get("delimiter", ","))
        elif suffix == ".parquet":
            df.write_parquet(path, compression=opts.get("compression", "snappy"))
        elif suffix in (".xlsx",):
            df.write_excel(path)
        elif suffix in (".json", ".ndjson"):
            df.write_ndjson(path)
        else:
            raise ValueError(f"Unsupported write format: {suffix}")
        return WriteResult(rows_written=len(df), duration_s=time.perf_counter() - t0)