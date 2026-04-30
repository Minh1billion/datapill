from pathlib import Path
from typing import Optional
import polars as pl
import io

def resolve_path(base_path: Path, path: str) -> Path:
    p = (base_path / path).resolve()
    base = base_path.resolve()
    if base not in p.parents and p != base:
        raise ValueError("invalid path")
    return p

def resolve_key(prefix: Optional[str], path: str) -> str:
    return f"{prefix.rstrip('/')}/{path.lstrip('/')}" if prefix else path


def get_format(path: str, fmt: Optional[str]) -> str:
    return (fmt or path.rsplit(".", 1)[-1]).lower()


def parse_bytes(data: bytes, fmt: str, encoding: str = "utf-8") -> pl.DataFrame:
    if fmt == "parquet":
        return pl.read_parquet(io.BytesIO(data))
    if fmt == "csv":
        return pl.read_csv(io.BytesIO(data), encoding=encoding)
    if fmt in ("json", "ndjson"):
        return pl.read_ndjson(io.BytesIO(data))
    raise ValueError(f"unsupported format: {fmt}")