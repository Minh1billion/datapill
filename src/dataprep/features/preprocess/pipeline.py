import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Any

import polars as pl

from dataprep.core.context import PipelineContext
from dataprep.core.events import EventType, ProgressEvent
from dataprep.core.interfaces import ExecutionPlan, FeaturePipeline, ValidationResult
from .schema import RunReport, StepConfig, StepResult
from .steps import build_step

_DRY_RUN_ROWS = 1000
_PREVIEW_ROWS = 10

_NUMERIC_STEPS = {
    "impute_mean", "impute_median", "clip_iqr", "clip_zscore",
    "standard_scaler", "minmax_scaler", "robust_scaler",
}


class PreprocessPipeline(FeaturePipeline):
    def __init__(self, steps: list[StepConfig], checkpoint: bool = False) -> None:
        self.steps = steps
        self.checkpoint = checkpoint
        self.run_id = uuid.uuid4().hex[:8]

    def validate(self, context: PipelineContext) -> ValidationResult:
        errors: list[str] = []
        if not self.steps:
            errors.append("steps must not be empty")
        return ValidationResult(ok=len(errors) == 0, errors=errors)

    def plan(self, context: PipelineContext) -> ExecutionPlan:
        return ExecutionPlan(
            steps=[{"name": cfg.step, "columns": cfg.columns} for cfg in self.steps],
            metadata={"checkpoint": self.checkpoint},
        )

    async def execute(
        self, plan: ExecutionPlan, context: PipelineContext
    ) -> AsyncGenerator[ProgressEvent, None]:
        t0 = time.perf_counter()
        dry_run: bool = plan.metadata.get("dry_run", False)

        yield ProgressEvent(EventType.STARTED, "Preprocess started", progress_pct=0.0)

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
            f"Loaded {len(df):,} rows, {len(df.columns)} columns",
            progress_pct=0.05,
        )

        try:
            report, transformed_df = self.run(df, dry_run=dry_run)
        except Exception as exc:
            yield ProgressEvent(EventType.ERROR, str(exc))
            raise

        for step_index, checkpoint_df in enumerate(report.checkpoints):
            checkpoint_id = f"{self.run_id}_checkpoint_step_{step_index}"
            await context.artifact_store.save_parquet(checkpoint_id, checkpoint_df)

        output_id = f"{self.run_id}_preprocess_output"
        await context.artifact_store.save_parquet(output_id, transformed_df)

        config_id = f"{self.run_id}_preprocess_config"
        await context.artifact_store.save_json(config_id, self.serialize())

        duration = time.perf_counter() - t0
        yield ProgressEvent(
            EventType.DONE,
            f"Preprocess complete in {duration:.1f}s — {len(self.steps)} steps",
            progress_pct=1.0,
            payload={
                "run_id": report.run_id,
                "dry_run": report.dry_run,
                "output_artifact_id": output_id,
                "config_artifact_id": config_id,
                "duration_s": round(duration, 3),
            },
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "feature": "preprocess",
            "steps": [
                {
                    "id": f"step_{i:03d}",
                    "type": cfg.step,
                    "scope": {"columns": cfg.columns},
                }
                for i, cfg in enumerate(self.steps)
            ],
            "checkpoint": self.checkpoint,
        }

    def run(self, df: pl.DataFrame, dry_run: bool = False) -> tuple[RunReport, pl.DataFrame]:
        warnings = self._detect_conflicts()
        source = df.head(_DRY_RUN_ROWS) if dry_run else df

        results: list[StepResult] = []
        checkpoints: list[pl.DataFrame] = []
        current = source

        for n, cfg in enumerate(self.steps):
            step = build_step(cfg)
            self._validate_numeric_requirement(cfg, current)
            current, result = step.apply(current)
            results.append(result)

            if self.checkpoint and not dry_run:
                checkpoints.append(current)

        report = RunReport(
            run_id=self.run_id,
            dry_run=dry_run,
            steps=results,
            warnings=warnings,
        )
        report.checkpoints = checkpoints

        if dry_run:
            report.preview_rows = current.head(_PREVIEW_ROWS).to_dicts()
            report.output_schema = {
                col: str(dtype) for col, dtype in zip(current.columns, current.dtypes)
            }

        return report, current

    def resume(self, df: pl.DataFrame, from_step: int) -> tuple[RunReport, pl.DataFrame]:
        parent_run_id = self.run_id
        partial_pipeline = PreprocessPipeline(
            steps=self.steps[from_step:],
            checkpoint=self.checkpoint,
        )

        partial_pipeline.run_id = parent_run_id
        return partial_pipeline.run(df)

    def _detect_conflicts(self) -> list[str]:
        warnings: list[str] = []
        imputed: dict[str, int] = {}
        iqr_cols: dict[str, int] = {}

        for i, cfg in enumerate(self.steps):
            scope = set(cfg.columns)

            if cfg.step in ("impute_mean", "impute_median", "impute_mode"):
                for col in scope:
                    imputed[col] = i

            if cfg.step == "drop_missing":
                for col in scope:
                    if col in imputed:
                        warnings.append(
                            f"Step {i} 'drop_missing' on column '{col}' is redundant: "
                            f"column was already imputed at step {imputed[col]}."
                        )

            if cfg.step == "clip_iqr":
                for col in scope:
                    iqr_cols[col] = i

            if cfg.step == "clip_zscore":
                for col in scope:
                    if col in iqr_cols:
                        warnings.append(
                            f"Step {i} 'clip_zscore' on column '{col}' overlaps with "
                            f"'clip_iqr' at step {iqr_cols[col]}."
                        )

        return warnings

    def _validate_numeric_requirement(self, cfg: StepConfig, df: pl.DataFrame) -> None:
        if cfg.step not in _NUMERIC_STEPS:
            return
        cols = cfg.columns or df.columns
        for col in cols:
            if col in df.columns and not df[col].dtype.is_numeric():
                raise TypeError(
                    f"Step '{cfg.step}' requires numeric column, "
                    f"but '{col}' is {df[col].dtype}."
                )