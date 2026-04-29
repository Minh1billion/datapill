import time
import uuid
from typing import AsyncGenerator, Any

import polars as pl

from datapill.core.context import PipelineContext
from datapill.core.events import EventType, ProgressEvent
from datapill.core.interfaces import ExecutionPlan, FeaturePipeline, ValidationResult
from .schema import ClassifyConfig, ClassifyResult, ProfileSignals, extract_profile_signals
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
        steps = ["load_data", "load_profile", "classify_columns", "build_output"]
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
        df: pl.DataFrame | None = None

        if input_ref and context.artifact_store.is_ref(input_ref):
            ref = await context.artifact_store.load_ref(input_ref)
            schema = ref.get("schema", [])
            if not schema:
                yield ProgressEvent(EventType.ERROR, "Ref artifact has no schema - cannot classify")
                return

            yield ProgressEvent(
                EventType.PROGRESS,
                "Using schema from ref artifact - no re-stream needed",
                progress_pct=0.05,
            )

            data = {
                col["name"]: pl.Series(
                    col["name"],
                    [],
                    dtype=_polars_dtype_from_str(col["dtype"]),
                )
                for col in schema
            }
            df = pl.DataFrame(data)

        elif input_ref:
            df = context.artifact_store.scan_parquet(input_ref).collect()
        elif "dataframe" in plan.metadata:
            df = plan.metadata["dataframe"]
        else:
            yield ProgressEvent(EventType.ERROR, "No input data provided")
            return

        profile_signals_map: dict[str, ProfileSignals] | None = None
        profile_artifact_id = plan.metadata.get("profile_artifact_id")

        if profile_artifact_id:
            try:
                resolved_profile_id = context.artifact_store.resolve(
                    profile_artifact_id, feature_hint="profile_detail"
                )
                profile_data = await context.artifact_store.load_json(resolved_profile_id)
                profile_signals_map = extract_profile_signals(profile_data)
                yield ProgressEvent(
                    EventType.PROGRESS,
                    f"Loaded profile signals for {len(profile_signals_map)} columns",
                    progress_pct=0.10,
                )
            except Exception as exc:
                yield ProgressEvent(
                    EventType.PROGRESS,
                    f"Warning: could not load profile artifact '{profile_artifact_id}': {exc} - continuing without profile",
                    progress_pct=0.10,
                )

        yield ProgressEvent(
            EventType.PROGRESS,
            f"Classifying {len(df.columns)} columns - mode: {self.config.mode}"
            + (" (with profile)" if profile_signals_map else ""),
            progress_pct=0.15,
        )

        try:
            result = run_classify(df, self.config, profile_signals_map=profile_signals_map)
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
                "profile_used": result.profile_used,
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

    def run(self, df: pl.DataFrame, profile_signals_map: dict[str, ProfileSignals] | None = None) -> ClassifyResult:
        return run_classify(df, self.config, profile_signals_map=profile_signals_map)


def _polars_dtype_from_str(dtype_str: str) -> pl.DataType:
    _MAP = {
        "Int8": pl.Int8,
        "Int16": pl.Int16,
        "Int32": pl.Int32,
        "Int64": pl.Int64,
        "UInt8": pl.UInt8,
        "UInt16": pl.UInt16,
        "UInt32": pl.UInt32,
        "UInt64": pl.UInt64,
        "Float32": pl.Float32,
        "Float64": pl.Float64,
        "Boolean": pl.Boolean,
        "String": pl.String,
        "Utf8": pl.Utf8,
        "Date": pl.Date,
        "Datetime": pl.Datetime,
        "Duration": pl.Duration,
        "Time": pl.Time,
        "Binary": pl.Binary,
        "Null": pl.Null,
    }
    for key, dtype in _MAP.items():
        if dtype_str.startswith(key):
            return dtype
    return pl.Utf8