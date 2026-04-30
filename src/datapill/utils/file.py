from pathlib import Path
import polars as pl

def resolve_path(base_path: Path, path: str) -> Path:
    p = (base_path / path).resolve()
    base = base_path.resolve()
    if base not in p.parents and p != base:
        raise ValueError("invalid path")
    return p