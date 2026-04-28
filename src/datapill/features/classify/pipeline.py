import time
import uuid
from typing import AsyncGenerator, Any

import polars as pl

from datapill.core.context import PipelineContext
from datapill.core.events import EventType, ProgressEvent
from datapill.core.interfaces import ExecutionPlan, FeaturePipeline, ValidationResult
from .schema import ClassifyConfig, ClassifyResult
from .runner import run_classify


class ClassifyPipeline(FeaturePipeline):
    def __init__(self, config: ClassifyConfig) -> None:
        self.config = config
        self.run_id = uuid.uuid4().hex[:8]

    def validate(self, context: PipelineContext) -> ValidationResult:
        errors: list[str] = []
        if self.config.mode not in ("rule_based", "embedding", "hybrid"):
            errors.append(f"Invalid mode: {self.config.mode!r}. Must be rule_based | embedding | hybrid")
        if self.config.confidence_threshold < 0.0 or self.config.confidence_threshold > 1.0:
            errors.append("confidence_threshold must be between 0.0 and 1.0")
        return ValidationResult(ok=len(errors) == 0, errors=errors)

    def plan(self, context: PipelineContext) -> ExecutionPlan:
        steps = ["load_data", "classify_columns", "build_output"]
        return ExecutionPlan(
            steps=[{"name": s} for s in steps],
            metadata={
                "mode": self.config.mode,
                "confidence_threshold": self.config.confidence_threshold,
            },
        )

    async def execute(
        self, plan: ExecutionPlan, context: PipelineContext
    ) -> AsyncGenerator[ProgressEvent, None]:
        t0 = time.perf_counter()

        yield ProgressEvent(EventType.STARTED, "Classify started", progress_pct=0.0)

        input_ref = plan.metadata.get("input_artifact_id")
        if input_ref:
            df = context.artifact_store.scan_parquet(input_ref).collect()
        elif "dataframe" in plan.metadata:
            df = plan.metadata["dataframe"]
        else:
            yield ProgressEvent(EventType.ERROR, "No input data provided")
            return

        yield ProgressEvent(
            EventType.PROGRESS,
            f"Loaded {len(df):,} rows, {len(df.columns)} columns - running {self.config.mode} classifier",
            progress_pct=0.1,
        )

        try:
            result = run_classify(df, self.config)
        except Exception as exc:
            yield ProgressEvent(EventType.ERROR, str(exc))
            raise

        yield ProgressEvent(EventType.PROGRESS, "Saving results", progress_pct=0.85)

        output_id = f"{self.run_id}_classify_output"
        await context.artifact_store.save_json(output_id, result.to_dict(), feature="classify")

        duration = time.perf_counter() - t0
        yield ProgressEvent(
            EventType.DONE,
            f"Classify complete in {duration:.1f}s - {len(result.columns)} columns classified",
            progress_pct=1.0,
            payload={
                "run_id": self.run_id,
                "output_artifact_id": output_id,
                "column_count": len(result.columns),
                "duration_s": round(duration, 3),
            },
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "feature": "classify",
            "mode": self.config.mode,
            "confidence_threshold": self.config.confidence_threshold,
            "overrides": self.config.overrides,
        }

    def run(self, df: pl.DataFrame) -> ClassifyResult:
        return run_classify(df, self.config)