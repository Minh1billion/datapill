from __future__ import annotations

import json
import shutil
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
            run_id=str(uuid.uuid4())[:8],
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


def _row_to_artifact(row: tuple) -> Artifact:
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
        return _row_to_artifact(row) if row else None

    def last(self, pipeline: Optional[str] = None) -> Optional[Artifact]:
        if pipeline:
            row = self._db.execute(
                "SELECT * FROM artifacts WHERE pipeline = ? ORDER BY timestamp DESC LIMIT 1",
                [pipeline],
            ).fetchone()
        else:
            row = self._db.execute(
                "SELECT * FROM artifacts ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return _row_to_artifact(row) if row else None

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
        return [_row_to_artifact(r) for r in rows]

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
        return [_row_to_artifact(r) for r in rows]

    def delete(self, run_id: str, cascade: bool = False) -> bool:
        children = self._db.execute(
            "SELECT run_id FROM artifacts WHERE parent_run_id = ?", [run_id]
        ).fetchall()

        if children and not cascade:
            child_ids = ", ".join(r[0] for r in children)
            raise RuntimeError(
                f"cannot delete {run_id}: has {len(children)} child artifact(s): {child_ids}"
            )

        if cascade:
            subtree = self.lineage(run_id)
            for artifact in reversed(subtree):
                if artifact.run_id != run_id:
                    self.delete(artifact.run_id, cascade=False)

        row = self._db.execute(
            "SELECT materialized, path FROM artifacts WHERE run_id = ?", [run_id]
        ).fetchone()
        if not row:
            return False
        materialized, rel_path = row
        if materialized and rel_path:
            abs_path = self.path / rel_path
            if abs_path.exists():
                if abs_path.is_dir():
                    shutil.rmtree(abs_path)
                else:
                    abs_path.unlink()
                parent = abs_path.parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
        self._db.execute("DELETE FROM artifacts WHERE run_id = ?", [run_id])
        return True

    def purge(
        self,
        pipeline: Optional[str] = None,
        keep: int = 0,
        only_samples: bool = False,
    ) -> list[str]:
        filters, params = ["parent_run_id IS NULL"], []
        if pipeline:
            filters.append("pipeline = ?")
            params.append(pipeline)
        where = "WHERE " + " AND ".join(filters)

        roots = self._db.execute(
            f"SELECT run_id FROM artifacts {where} ORDER BY timestamp DESC", params
        ).fetchall()
        roots_to_delete = [r[0] for r in roots][keep:]

        deleted = []
        for root_id in roots_to_delete:
            subtree = self.lineage(root_id)
            for artifact in reversed(subtree):
                if only_samples and not artifact.is_sample:
                    continue
                if self.delete(artifact.run_id):
                    deleted.append(artifact.run_id)

        return deleted

    def disk_usage(self) -> int:
        artifacts_dir = self.path / "artifacts"
        if not artifacts_dir.exists():
            return 0
        return sum(f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file())

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "ArtifactStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()