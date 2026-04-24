import json
import hashlib
from pathlib import Path
from typing import Any
import aiofiles
import polars as pl


class ArtifactStore:
    def __init__(self, base_path: str = "/src/dataprep/artifacts") -> None:
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_id: str, ext: str) -> Path:
        return self.base / f"{artifact_id}.{ext}"

    async def save_json(self, artifact_id: str, data: dict[str, Any]) -> Path:
        path = self._path(artifact_id, "json")
        payload = json.dumps(data, indent=2, default=str)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(payload)
        return path

    async def load_json(self, artifact_id: str) -> dict[str, Any]:
        path = self._path(artifact_id, "json")
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return json.loads(await f.read())

    async def save_parquet(self, artifact_id: str, df: pl.DataFrame) -> Path:
        path = self._path(artifact_id, "parquet")
        df.write_parquet(path, compression="snappy")
        return path

    def load_parquet(self, artifact_id: str) -> pl.DataFrame:
        return pl.read_parquet(self._path(artifact_id, "parquet"))

    def scan_parquet(self, artifact_id: str) -> pl.LazyFrame:
        return pl.scan_parquet(self._path(artifact_id, "parquet"))

    def exists(self, artifact_id: str, ext: str = "json") -> bool:
        return self._path(artifact_id, ext).exists()

    def checksum(self, artifact_id: str, ext: str = "json") -> str:
        path = self._path(artifact_id, ext)
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()