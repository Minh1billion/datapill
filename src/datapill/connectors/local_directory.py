import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, AsyncGenerator, Any
import polars as pl

from .base import BaseConnector, ConnectionStatus
from ..utils.file import resolve_path
from ..utils.streaming import estimate_batch_size

@dataclass
class LocalConnectorConfig:
    base_path: str
    mkdir: bool = False
    read_only: bool = False
    format_filter: Optional[list[str]] = None
    encoding: str = "utf-8"
    recursive: bool = False

class LocalDirectoryConnector(BaseConnector[LocalConnectorConfig]):
    def __init__(self, config: LocalConnectorConfig):
        super().__init__(config)
        self.base_path = Path(config.base_path)

    async def connect(self) -> ConnectionStatus:
        t0 = time.perf_counter()

        if self.config.mkdir:
            self.base_path.mkdir(parents=True, exist_ok=True)

        if not self.base_path.exists():
            return ConnectionStatus(ok=False, error=f"{self.base_path} does not exist")

        if not self.base_path.is_dir():
            return ConnectionStatus(ok=False, error=f"{self.base_path} is not a directory")

        return ConnectionStatus(ok=True, latency_ms=1000 * (time.perf_counter() - t0))

    async def cleanup(self) -> None:
        return

    async def read(
        self,
        path: str,
        format: Optional[str] = None,
        stream: bool = False,
        batch_size: Optional[int] = None,
    ) -> pl.DataFrame | AsyncGenerator[pl.DataFrame, Any]:
        p = resolve_path(self.base_path, path)

        if format is None:
            format = p.suffix.lstrip(".")

        if self.config.format_filter and format not in self.config.format_filter:
            raise ValueError("format not allowed")

        if not stream:
            if format == "csv":
                return pl.read_csv(p, encoding=self.config.encoding)
            if format == "parquet":
                return pl.read_parquet(p)
            if format in ("json", "ndjson"):
                return pl.read_json(p)
            raise ValueError(f"unsupported format: {format}")

        file_size = p.stat().st_size

        if format == "parquet":
            meta = pl.read_parquet_metadata(p)
            batch_size = estimate_batch_size(file_size, meta.num_rows) if batch_size is None and stream else batch_size
            return pl.scan_parquet(p).collect(streaming=True).iter_slices(batch_size)

        if format == "csv":
            sample = pl.read_csv(p, n_rows=1000, encoding=self.config.encoding)
            batch_size = estimate_batch_size(file_size, len(sample)) if batch_size is None and stream else batch_size
            return pl.scan_csv(p).collect_batches(batch_size)

        if format in ("json", "ndjson"):
            sample = pl.read_ndjson(p, n_rows=1000)
            batch_size = estimate_batch_size(file_size, len(sample)) if batch_size is None and stream else batch_size
            return pl.scan_ndjson(p).collect(streaming=True).iter_slices(batch_size)

        raise ValueError(f"unsupported format: {format}")

    async def write(
        self,
        df: pl.DataFrame,
        path: str,
        *,
        format: Optional[str] = None,
        mode: str = "overwrite"
    ) -> None:
        if self.config.read_only:
            raise PermissionError("read only")

        p = resolve_path(self.base_path, path)

        if format is None:
            format = p.suffix.replace(".", "")

        if self.config.format_filter and format not in self.config.format_filter:
            raise ValueError("format not allowed")

        if mode == "overwrite" and p.exists():
            p.unlink()

        if format == "csv":
            df.write_csv(p)
            return
        if format == "parquet":
            df.write_parquet(p)
            return
        if format == "json":
            df.write_json(p)
            return

        raise ValueError("unsupported format")