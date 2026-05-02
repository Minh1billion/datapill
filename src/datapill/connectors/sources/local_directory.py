import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, AsyncGenerator, Any

import polars as pl
import pyarrow.parquet as pq
import pyarrow as pa

from ..base import BaseConnector, ConnectionStatus
from ...utils.connection import timed_connect
from ...utils.file_process import resolve_path
from ...utils.streaming import estimate_batch_size


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
        fmt = format or p.suffix.lstrip(".")

        if self.config.format_filter and fmt not in self.config.format_filter:
            raise ValueError("format not allowed")

        if not stream:
            if fmt == "parquet":
                return await asyncio.to_thread(pl.read_parquet, p)
            if fmt == "csv":
                return await asyncio.to_thread(pl.read_csv, p, encoding=self.config.encoding)
            if fmt in ("json", "ndjson"):
                return await asyncio.to_thread(pl.read_ndjson, p)
            raise ValueError(f"unsupported format: {fmt}")

        async def generate() -> AsyncGenerator[pl.DataFrame, Any]:
            if fmt == "parquet":
                pf = await asyncio.to_thread(pq.ParquetFile, p)
                nb = batch_size or pf.metadata.row_group(0).num_rows
                for batch in pf.iter_batches(batch_size=nb):
                    yield await asyncio.to_thread(pl.from_arrow, batch)
            elif fmt == "csv":
                file_size = p.stat().st_size
                nb = batch_size or max(10_000, file_size // 200)
                _enc = "utf8" if self.config.encoding.lower() in ("utf-8", "utf8") else self.config.encoding
                reader = await asyncio.to_thread(
                    lambda: pl.read_csv_batched(p, batch_size=nb, encoding=_enc)
                )
                while True:
                    batches = await asyncio.to_thread(reader.next_batches, 1)
                    if not batches:
                        break
                    yield batches[0]
            elif fmt in ("json", "ndjson"):
                file_size = p.stat().st_size
                full = await asyncio.to_thread(pl.read_ndjson, p)
                nb = batch_size or estimate_batch_size(file_size, len(full))
                for i in range(0, len(full), nb):
                    yield full.slice(i, nb)
            else:
                raise ValueError(f"unsupported format: {fmt}")

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
        fmt = format or p.suffix.replace(".", "")

        if self.config.format_filter and fmt not in self.config.format_filter:
            raise ValueError("format not allowed")

        if mode == "overwrite" and p.exists():
            p.unlink()

        if fmt == "csv":
            await asyncio.to_thread(df.write_csv, p)
        elif fmt == "parquet":
            await asyncio.to_thread(df.write_parquet, p)
        elif fmt == "json":
            await asyncio.to_thread(df.write_json, p)
        else:
            raise ValueError("unsupported format")