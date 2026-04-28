import json
import hashlib
import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import aiofiles
import polars as pl

_REGISTRY_FILE = "registry.json"

_FEATURE_PRIORITY = {
    "profile":    ["ingest_output", "preprocess_output"],
    "preprocess": ["ingest_output", "preprocess_output"],
    "export":     ["preprocess_output", "ingest_output"],
    "classify":   ["ingest_output", "preprocess_output"],
}

class ArtifactStore:
    def __init__(self, base_path: str = "/src/dataprep/artifacts") -> None:
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, dict[str, Any]] = self._load_registry()

    def _registry_path(self) -> Path:
        return self.base / _REGISTRY_FILE

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        p = self._registry_path()
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_registry(self) -> None:
        with self._registry_path().open("w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def _register(self, artifact_id: str, feature: str, ext: str) -> None:
        parts = artifact_id.split("_", 1)
        run_id = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        self._registry[artifact_id] = {
            "run_id": run_id,
            "suffix": suffix,
            "feature": feature,
            "ext": ext,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_registry()

    def resolve(self, input_str: str, feature_hint: str = "") -> str:
        if input_str in self._registry:
            return input_str

        matches = {
            aid: meta
            for aid, meta in self._registry.items()
            if meta["run_id"] == input_str
        }

        if not matches:
            raise FileNotFoundError(
                f"No artifact found for '{input_str}'. "
                f"Use 'dp list' to see available artifacts."
            )

        if len(matches) == 1:
            return next(iter(matches))

        priority = _FEATURE_PRIORITY.get(feature_hint, [])
        for suffix in priority:
            for aid, meta in matches.items():
                if meta["suffix"] == suffix:
                    return aid

        latest = max(matches.items(), key=lambda x: x[1]["created_at"])
        return latest[0]

    def _path(self, artifact_id: str, ext: str) -> Path:
        return self.base / f"{artifact_id}.{ext}"

    async def save_json(self, artifact_id: str, data: dict[str, Any], feature: str = "") -> Path:
        path = self._path(artifact_id, "json")
        payload = json.dumps(data, indent=2, default=str)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(payload)
        self._register(artifact_id, feature, "json")
        return path

    async def load_json(self, artifact_id: str) -> dict[str, Any]:
        path = self._path(artifact_id, "json")
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return json.loads(await f.read())

    async def save_parquet(self, artifact_id: str, df: pl.DataFrame, feature: str = "") -> Path:
        path = self._path(artifact_id, "parquet")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: df.write_parquet(path, compression="snappy"))
        self._register(artifact_id, feature, "parquet")
        return path

    async def save_parquet_from_path(self, artifact_id: str, src: Path, feature: str = "") -> Path:
        dest = self._path(artifact_id, "parquet")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.move, str(src), str(dest))
        self._register(artifact_id, feature, "parquet")
        return dest

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

    def list_artifacts(self) -> list[dict[str, Any]]:
        return [
            {"artifact_id": aid, **meta}
            for aid, meta in sorted(
                self._registry.items(),
                key=lambda x: x[1]["created_at"],
                reverse=True,
            )
        ]