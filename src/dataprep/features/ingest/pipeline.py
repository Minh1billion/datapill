import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Any
import polars as pl
from dataprep.core.interfaces import FeaturePipeline, ValidationResult, ExecutionPlan
from dataprep.core.context import PipelineContext
from dataprep.core.events import ProgressEvent, EventType
from dataprep.connectors.base import BaseConnector


@dataclass
class IngestConfig:
    connector: BaseConnector
    query: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)


class IngestPipeline(FeaturePipeline):
    def __init__(self, config: IngestConfig) -> None:
        self.config = config

    def validate(self, context: PipelineContext) -> ValidationResult:
        errors: list[str] = []
        if not self.config.query:
            errors.append("query must not be empty")
        if not self.config.connector:
            errors.append("connector is required")
        return ValidationResult(ok=len(errors) == 0, errors=errors)

    def plan(self, context: PipelineContext) -> ExecutionPlan:
        return ExecutionPlan(
            steps=[{"name": "stream_read"}, {"name": "materialize_parquet"}, {"name": "save_schema"}],
            metadata={"query": self.config.query, "options": self.config.options},
        )

    async def execute(
        self, plan: ExecutionPlan, context: PipelineContext
    ) -> AsyncGenerator[ProgressEvent, None]:
        run_id = context.run_id
        t0 = time.perf_counter()
        rows_read = 0
        schema_snapshot: list[dict] | None = None
        chunks: list[pl.DataFrame] = []

        yield ProgressEvent(EventType.STARTED, "Ingest started", progress_pct=0.0)

        try:
            async for chunk in self.config.connector.read_stream(
                self.config.query, self.config.options
            ):
                if schema_snapshot is None:
                    schema_snapshot = _extract_schema(chunk)

                rows_read += len(chunk)
                chunks.append(chunk)

                yield ProgressEvent(
                    EventType.PROGRESS,
                    f"Read {rows_read:,} rows",
                    payload={"rows_read": rows_read},
                )

            df = pl.concat(chunks, rechunk=True) if chunks else pl.DataFrame()
            output_id = f"{run_id}_ingest_output"
            await context.artifact_store.save_parquet(output_id, df)

            schema_id = f"{run_id}_ingest_schema"
            duration = time.perf_counter() - t0
            await context.artifact_store.save_json(schema_id, {
                "schema": schema_snapshot or [],
                "row_count": rows_read,
                "column_count": len(df.columns),
                "ingest_duration_s": round(duration, 3),
                "query": self.config.query,
            })

            yield ProgressEvent(
                EventType.DONE,
                f"Ingest complete: {rows_read:,} rows in {duration:.1f}s",
                progress_pct=1.0,
                payload={
                    "output_artifact_id": output_id,
                    "schema_artifact_id": schema_id,
                    "rows_read": rows_read,
                },
            )

        except Exception as exc:
            yield ProgressEvent(EventType.ERROR, str(exc), payload={"rows_read": rows_read})
            raise

    def serialize(self) -> dict[str, Any]:
        return {
            "feature": "ingest",
            "version": "1.0",
            "query": self.config.query,
            "options": self.config.options,
        }


def _extract_schema(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": col,
            "dtype": str(df[col].dtype),
            "nullable": df[col].null_count() > 0,
        }
        for col in df.columns
    ]