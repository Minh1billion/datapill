from __future__ import annotations

from typing import Any, AsyncGenerator

import polars as pl

from ...connectors import registry
from ...core.context import Context
from ...core.events import EventType, ProgressEvent
from ...storage.artifact_store import Artifact
from ..base import ExecutionPlan, Pipeline, ValidationResult
from .readers import get_reader


class IngestPipeline(Pipeline):
    def __init__(self, source: str, config: dict[str, Any], options: dict[str, Any]) -> None:
        self.source = source
        self.config = config
        self.options = options

    def validate(self, context: Context) -> ValidationResult:
        errors: list[str] = []

        if self.source not in registry.sources():
            errors.append(f"unknown source: {self.source!r}")

        if self.source in {"postgres", "mysql", "sqlite"}:
            if not self.options.get("table") and not self.options.get("query"):
                errors.append("options must include 'table' or 'query'")

        if self.source == "kafka" and not self.options.get("topic"):
            errors.append("options must include 'topic'")

        if self.source in {"s3", "local"} and not self.options.get("path"):
            errors.append("options must include 'path'")

        return ValidationResult(ok=not errors, errors=errors)

    def plan(self, context: Context) -> ExecutionPlan:
        safe = registry.safe_config(self.source, self.config)
        is_sample = self.options.get("sample", False)
        sample_size = self.options.get("sample_size", 10_000)
        materialized = self.options.get("materialized", False)

        steps = [
            {"action": "test_connection", "source": self.source, **safe},
            {
                "action": "read",
                "mode": "sample" if is_sample else "full",
                **({"sample_size": sample_size} if is_sample else {}),
                **{k: v for k, v in self.options.items() if k in ("table", "query", "topic", "path")},
            },
        ]

        if materialized:
            steps.append({
                "action": "materialize",
                "format": "parquet",
                "path": f".datapill/artifacts/{context.run_id}/data.parquet",
            })

        return ExecutionPlan(
            steps=steps,
            metadata={"source": self.source, "safe_config": safe, "options": self.options},
        )

    async def execute(
        self, plan: ExecutionPlan, context: Context
    ) -> AsyncGenerator[ProgressEvent, None]:
        artifact = Artifact.new(
            pipeline="ingest",
            options={
                "source": self.source,
                **{k: v for k, v in self.options.items()
                   if k in ("table", "query", "topic", "path", "sample", "sample_size")},
            },
            is_sample=self.options.get("sample", False),
            sample_size=self.options.get("sample_size") if self.options.get("sample") else None,
        )

        yield ProgressEvent(event_type=EventType.STARTED, message=f"connecting to {self.source}")

        connector = registry.build(self.source, self.config)
        status = await connector.connect()

        if not status.ok:
            yield ProgressEvent(event_type=EventType.ERROR, message=f"connection failed: {status.error}")
            await connector.cleanup()
            return

        yield ProgressEvent(
            event_type=EventType.PROGRESS,
            message="connected",
            progress_pct=10.0,
            payload={"latency_ms": status.latency_ms},
        )

        try:
            reader = get_reader(self.source)
            df = await reader.read(
                connector=connector,
                options=self.options,
                is_sample=self.options.get("sample", False),
                sample_size=self.options.get("sample_size", 10_000),
            )
        except Exception as exc:
            yield ProgressEvent(event_type=EventType.ERROR, message=str(exc))
            await connector.cleanup()
            return
        finally:
            await connector.cleanup()

        artifact.schema = {name: str(dtype) for name, dtype in zip(df.columns, df.dtypes)}

        yield ProgressEvent(
            event_type=EventType.PROGRESS,
            message=f"read {len(df):,} rows",
            progress_pct=80.0,
            payload={"rows": len(df), "columns": len(df.columns)},
        )

        if self.options.get("materialized"):
            out = context.artifact_store.path / "artifacts" / artifact.run_id / "data.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out)
            artifact.materialized = True
            artifact.path = str(out.relative_to(context.artifact_store.path))
            yield ProgressEvent(
                event_type=EventType.PROGRESS,
                message=f"materialized to {artifact.path}",
                progress_pct=95.0,
                payload={"path": artifact.path},
            )

        context.artifact_store.save(artifact)
        context.artifact = artifact

        yield ProgressEvent(
            event_type=EventType.DONE,
            message="ingest complete",
            progress_pct=100.0,
            payload={"run_id": artifact.run_id, "ref": artifact.ref},
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "pipeline": "ingest",
            "version": "1.0",
            "source": self.source,
            "config": registry.safe_config(self.source, self.config),
            "options": self.options,
            "schema": {"input": None, "output": "parquet | none"},
            "capabilities": ["sample", "materialize", "stream"],
        }