from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb


@dataclass
class Artifact:
    run_id: str
    pipeline: str
    parent_run_id: Optional[str]
    timestamp: datetime

    is_sample: bool = False
    sample_size: Optional[int] = None
    options: dict[str, Any] = field(default_factory=dict)
    schema: dict[str, str] = field(default_factory=dict)

    materialized: bool = False
    path: Optional[str] = None

    @classmethod
    def new(
        cls,
        pipeline: str,
        parent: Optional["Artifact"] = None,
        options: Optional[dict[str, Any]] = None,
        is_sample: bool = False,
        sample_size: Optional[int] = None,
    ) -> "Artifact":
        return cls(
            run_id=str(uuid.uuid4()),
            pipeline=pipeline,
            parent_run_id=parent.run_id if parent else None,
            timestamp=datetime.now(timezone.utc),
            is_sample=is_sample,
            sample_size=sample_size,
            options=options or {},
        )

    @property
    def ref(self) -> str:
        return f"{self.pipeline}:{self.run_id}"

    def __str__(self) -> str:
        return self.ref

    def __repr__(self) -> str:
        parent = f" <- {self.parent_run_id[:8]}" if self.parent_run_id else ""
        return f"Artifact({self.pipeline}:{self.run_id[:8]}{parent})"


class ArtifactStore:
    def __init__(self, path: Path | str = ".datapill") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._db = duckdb.connect(str(self.path / "store.db"))
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                run_id          TEXT PRIMARY KEY,
                pipeline        TEXT NOT NULL,
                parent_run_id   TEXT,
                timestamp       TIMESTAMPTZ NOT NULL,
                is_sample       BOOLEAN NOT NULL DEFAULT FALSE,
                sample_size     INTEGER,
                options         TEXT NOT NULL DEFAULT '{}',
                schema          TEXT NOT NULL DEFAULT '{}',
                materialized    BOOLEAN NOT NULL DEFAULT FALSE,
                path            TEXT
            )
        """)

    def save(self, artifact: Artifact) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                artifact.run_id,
                artifact.pipeline,
                artifact.parent_run_id,
                artifact.timestamp,
                artifact.is_sample,
                artifact.sample_size,
                json.dumps(artifact.options),
                json.dumps(artifact.schema),
                artifact.materialized,
                artifact.path,
            ],
        )

    def get(self, run_id: str) -> Optional[Artifact]:
        row = self._db.execute(
            "SELECT * FROM artifacts WHERE run_id = ?", [run_id]
        ).fetchone()
        if not row:
            return None
        run_id, pipeline, parent_run_id, timestamp, is_sample, sample_size, options, schema, materialized, path = row
        return Artifact(
            run_id=run_id,
            pipeline=pipeline,
            parent_run_id=parent_run_id,
            timestamp=timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp)),
            is_sample=bool(is_sample),
            sample_size=sample_size,
            options=json.loads(options),
            schema=json.loads(schema),
            materialized=bool(materialized),
            path=path,
        )

    def last(self, pipeline: Optional[str] = None) -> Optional[Artifact]:
        q = "SELECT * FROM artifacts ORDER BY timestamp DESC LIMIT 1"
        if pipeline:
            q = "SELECT * FROM artifacts WHERE pipeline = ? ORDER BY timestamp DESC LIMIT 1"
        row = self._db.execute(q, [pipeline] if pipeline else []).fetchone()
        if not row:
            return None
        run_id, pipeline, parent_run_id, timestamp, is_sample, sample_size, options, schema, materialized, path = row
        return Artifact(
            run_id=run_id,
            pipeline=pipeline,
            parent_run_id=parent_run_id,
            timestamp=timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp)),
            is_sample=bool(is_sample),
            sample_size=sample_size,
            options=json.loads(options),
            schema=json.loads(schema),
            materialized=bool(materialized),
            path=path,
        )

    def list(self, pipeline: Optional[str] = None, limit: int = 20) -> list[Artifact]:
        if pipeline:
            rows = self._db.execute(
                "SELECT * FROM artifacts WHERE pipeline = ? ORDER BY timestamp DESC LIMIT ?",
                [pipeline, limit],
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM artifacts ORDER BY timestamp DESC LIMIT ?", [limit]
            ).fetchall()
        result = []
        for run_id, pipeline, parent_run_id, timestamp, is_sample, sample_size, options, schema, materialized, path in rows:
            result.append(Artifact(
                run_id=run_id,
                pipeline=pipeline,
                parent_run_id=parent_run_id,
                timestamp=timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp)),
                is_sample=bool(is_sample),
                sample_size=sample_size,
                options=json.loads(options),
                schema=json.loads(schema),
                materialized=bool(materialized),
                path=path,
            ))
        return result

    def lineage(self, run_id: str) -> list[Artifact]:
        rows = self._db.execute("""
            WITH RECURSIVE chain AS (
                SELECT * FROM artifacts WHERE run_id = ?
                UNION ALL
                SELECT a.* FROM artifacts a
                JOIN chain c ON a.run_id = c.parent_run_id
            )
            SELECT * FROM chain ORDER BY timestamp ASC
        """, [run_id]).fetchall()
        result = []
        for run_id, pipeline, parent_run_id, timestamp, is_sample, sample_size, options, schema, materialized, path in rows:
            result.append(Artifact(
                run_id=run_id,
                pipeline=pipeline,
                parent_run_id=parent_run_id,
                timestamp=timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp)),
                is_sample=bool(is_sample),
                sample_size=sample_size,
                options=json.loads(options),
                schema=json.loads(schema),
                materialized=bool(materialized),
                path=path,
            ))
        return result

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "ArtifactStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()