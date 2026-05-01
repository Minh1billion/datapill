import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, AsyncGenerator, Any

import polars as pl
import pyarrow.parquet as pq

from .base import BaseConnector, ConnectionStatus
from ..utils.connection import timed_connect
from ..utils.file_process import resolve_path
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
        async def probe():
            if self.config.mkdir:
                self.base_path.mkdir(parents=True, exist_ok=True)
            if not self.base_path.exists():
                raise FileNotFoundError(f"{self.base_path} does not exist")
            if not self.base_path.is_dir():
                raise NotADirectoryError(f"{self.base_path} is not a directory")

        ok, latency_ms, error = await timed_connect(probe)
        return ConnectionStatus(ok=ok, latency_ms=latency_ms, error=error)

    async def cleanup(self) -> None:
        return

    async def read(
        self,
        path: str,
        format: Optional[str] = None,
        stream: bool = False,
        batch_size: Optional[int] = None,
    ) -> "pl.DataFrame | AsyncGenerator[pl.DataFrame, Any]":
        p = resolve_path(self.base_path, path)

        if format is None:
            format = p.suffix.lstrip(".")

        if self.config.format_filter and format not in self.config.format_filter:
            raise ValueError("format not allowed")

        if not stream:
            if format == "parquet":
                return await asyncio.to_thread(pl.read_parquet, p)
            if format == "csv":
                return await asyncio.to_thread(pl.read_csv, p, encoding=self.config.encoding)
            if format in ("json", "ndjson"):
                return await asyncio.to_thread(pl.read_ndjson, p)
            raise ValueError(f"unsupported format: {format}")

        async def generate() -> AsyncGenerator[pl.DataFrame, Any]:
            if format == "parquet":
                pf = await asyncio.to_thread(pq.ParquetFile, p)
                for batch in pf.iter_batches():
                    yield await asyncio.to_thread(pl.from_arrow, batch)
            elif format == "csv":
                file_size = p.stat().st_size
                full = await asyncio.to_thread(pl.read_csv, p, encoding=self.config.encoding)
                resolved = batch_size or estimate_batch_size(file_size, len(full))
                for i in range(0, len(full), resolved):
                    yield full.slice(i, resolved)
            elif format in ("json", "ndjson"):
                file_size = p.stat().st_size
                full = await asyncio.to_thread(pl.read_ndjson, p)
                resolved = batch_size or estimate_batch_size(file_size, len(full))
                for i in range(0, len(full), resolved):
                    yield full.slice(i, resolved)
            else:
                raise ValueError(f"unsupported format: {format}")

        return generate()

    async def write(
        self,
        df: pl.DataFrame,
        path: str,
        *,
        format: Optional[str] = None,
        mode: str = "overwrite",
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
            await asyncio.to_thread(df.write_csv, p)
        elif format == "parquet":
            await asyncio.to_thread(df.write_parquet, p)
        elif format == "json":
            await asyncio.to_thread(df.write_json, p)
        else:
            raise ValueError("unsupported format")