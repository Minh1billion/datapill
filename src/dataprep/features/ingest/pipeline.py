import time
import uuid
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Any
 
import polars as pl
import pyarrow.parquet as pq
import pyarrow as pa
 
from dataprep.core.context import PipelineContext
from dataprep.core.events import EventType, ProgressEvent
from dataprep.core.interfaces import ExecutionPlan, FeaturePipeline, ValidationResult
from .schema import IngestConfig
 
 
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
            steps=[
                {"name": "stream_read"},
                {"name": "materialize_parquet"},
                {"name": "save_schema"},
            ],
            metadata={"query": self.config.query, "options": self.config.options},
        )
 
    async def execute(
        self, plan: ExecutionPlan, context: PipelineContext
    ) -> AsyncGenerator[ProgressEvent, None]:
        run_id = context.run_id
        t0 = time.perf_counter()
        rows_read = 0
        schema_snapshot: list[dict] | None = None

        tmp_path = Path(tempfile.mktemp(suffix=".parquet"))
        parquet_writer = None
        col_count = 0
 
        yield ProgressEvent(EventType.STARTED, "Ingest started", progress_pct=0.0)
 
        try:
            async for chunk in self.config.connector.read_stream(
                self.config.query, self.config.options
            ):
                if chunk.is_empty():
                    continue
 
                if schema_snapshot is None:
                    schema_snapshot = _extract_schema(chunk)
                    col_count = len(chunk.columns)
 
                rows_read += len(chunk)

                chunk.write_parquet(
                    tmp_path if parquet_writer is None else tmp_path,
                    compression="snappy",
                )
                parquet_writer = True
 
                yield ProgressEvent(
                    EventType.PROGRESS,
                    f"Read {rows_read:,} rows",
                    payload={"rows_read": rows_read},
                )
 
 
            writer: pq.ParquetWriter | None = None
            rows_read = 0
            schema_snapshot = None
 
            async for chunk in self.config.connector.read_stream(
                self.config.query, self.config.options
            ):
                if chunk.is_empty():
                    continue
                if schema_snapshot is None:
                    schema_snapshot = _extract_schema(chunk)
                    col_count = len(chunk.columns)
                rows_read += len(chunk)
                arrow_table = chunk.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(str(tmp_path), arrow_table.schema, compression="snappy")
                writer.write_table(arrow_table)
                yield ProgressEvent(
                    EventType.PROGRESS,
                    f"Read {rows_read:,} rows",
                    payload={"rows_read": rows_read},
                )
 
            if writer:
                writer.close()
 
            df = pl.read_parquet(tmp_path) if tmp_path.exists() else pl.DataFrame()
 
            output_id = f"{run_id}_ingest_output"
            await context.artifact_store.save_parquet(output_id, df)
 
            schema_id = f"{run_id}_ingest_schema"
            duration = time.perf_counter() - t0
            await context.artifact_store.save_json(schema_id, {
                "schema": schema_snapshot or [],
                "row_count": rows_read,
                "column_count": col_count,
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
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
 
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