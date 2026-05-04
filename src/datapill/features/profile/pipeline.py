import json
import random
from typing import Any, AsyncGenerator

import polars as pl

from ...core.context import Context
from ...core.events import EventType, ProgressEvent
from ...storage.artifact_store import Artifact
from ..base import ExecutionPlan, Pipeline, ValidationResult
from ...utils.loader import load_dataframe
from .stats import (
    ProfileOptions,
    collect_dataset_warnings,
    compute_column_profile,
    compute_correlations,
    compute_summary,
)

_VALID_PARENTS = {"ingest", "preprocess"}


class ProfilePipeline(Pipeline):
    def __init__(self, parent_run_id: str, options: ProfileOptions | None = None) -> None:
        self.parent_run_id = parent_run_id
        self.options = options or ProfileOptions()

    def validate(self, context: Context) -> ValidationResult:
        errors: list[str] = []

        parent = context.artifact_store.get(self.parent_run_id)
        if parent is None:
            errors.append(f"artifact not found: {self.parent_run_id!r}")
            return ValidationResult(ok=False, errors=errors)

        if parent.pipeline not in _VALID_PARENTS:
            errors.append(
                f"profile cannot accept input from {parent.pipeline!r}, "
                f"expected one of: {sorted(_VALID_PARENTS)}"
            )

        if self.options.mode not in ("full", "summary"):
            errors.append(f"invalid mode: {self.options.mode!r}, expected 'full' or 'summary'")

        if self.options.sample_strategy not in ("none", "random", "reservoir"):
            errors.append(f"invalid sample_strategy: {self.options.sample_strategy!r}")

        if self.options.correlation_method not in ("pearson", "spearman", "none"):
            errors.append(f"invalid correlation_method: {self.options.correlation_method!r}")

        return ValidationResult(ok=not errors, errors=errors)

    def plan(self, context: Context) -> ExecutionPlan:
        parent = context.artifact_store.get(self.parent_run_id)
        load_mode = "parquet" if (parent and parent.materialized) else "connector"

        steps: list[dict[str, Any]] = [
            {"action": "load_data", "mode": load_mode, "parent_run_id": self.parent_run_id},
            {"action": "sample", "strategy": self.options.sample_strategy},
            {"action": "compute_summary", "chunk_size": self.options.chunk_size},
            {"action": "profile_columns", "chunk_size": self.options.chunk_size},
        ]

        if self.options.mode == "full":
            steps.append({
                "action": "compute_correlations",
                "method": self.options.correlation_method,
                "threshold": self.options.correlation_threshold,
            })
            steps.append({"action": "materialize", "format": "json"})

        return ExecutionPlan(
            steps=steps,
            metadata={
                "parent_run_id": self.parent_run_id,
                "load_mode": load_mode,
                "options": {
                    "mode": self.options.mode,
                    "chunk_size": self.options.chunk_size,
                    "sample_strategy": self.options.sample_strategy,
                    "sample_size": self.options.sample_size,
                    "correlation_method": self.options.correlation_method,
                },
            },
        )

    async def execute(
        self, plan: ExecutionPlan, context: Context
    ) -> AsyncGenerator[ProgressEvent, None]:
        parent = context.artifact_store.get(self.parent_run_id)
        artifact = Artifact.new(
            pipeline="profile",
            parent=parent,
            options={
                "mode": self.options.mode,
                "sample_strategy": self.options.sample_strategy,
                "sample_size": self.options.sample_size,
                "correlation_method": self.options.correlation_method,
            },
            is_sample=parent.is_sample if parent else False,
            sample_size=parent.sample_size if parent else None,
        )

        yield ProgressEvent(event_type=EventType.STARTED, message="loading data from parent artifact")

        try:
            df = await load_dataframe(parent, context)
        except Exception as exc:
            yield ProgressEvent(event_type=EventType.ERROR, message=str(exc))
            return

        if self.options.sample_strategy == "random" and len(df) > self.options.sample_size:
            df = df.sample(n=self.options.sample_size, seed=42)
        elif self.options.sample_strategy == "reservoir" and len(df) > self.options.sample_size:
            k = self.options.sample_size
            indices = list(range(k))
            for i in range(k, len(df)):
                j = random.randint(0, i)
                if j < k:
                    indices[j] = i
            df = df[sorted(indices)]

        n_rows, n_cols = df.shape
        yield ProgressEvent(
            event_type=EventType.PROGRESS,
            message=f"loaded {n_rows:,} rows, {n_cols} columns",
            progress_pct=10.0,
            payload={"rows": n_rows, "columns": n_cols},
        )

        yield ProgressEvent(
            event_type=EventType.PROGRESS,
            message="computing dataset-level summary",
            progress_pct=15.0,
        )
        summary = compute_summary(df, self.options)

        column_profiles: list[dict[str, Any]] = []
        for i, col in enumerate(df.columns):
            try:
                col_profile = compute_column_profile(df, col, self.options)
            except Exception as exc:
                col_profile = {
                    "name": col,
                    "dtype_physical": str(df[col].dtype),
                    "dtype_inferred": "unknown",
                    "null_count": 0,
                    "null_pct": 0.0,
                    "distinct_count": 0,
                    "distinct_pct": 0.0,
                    "is_unique": False,
                    "min": None, "max": None, "mean": None, "median": None,
                    "std": None, "variance": None, "skewness": None, "kurtosis": None,
                    "percentiles": None, "histogram": None, "top_values": None,
                    "pattern_matches": None, "sample_values": [],
                    "warnings": [{"code": "PROFILE_ERROR", "column": col, "severity": "error", "detail": str(exc)}],
                }

            column_profiles.append(col_profile)
            pct = 20.0 + ((i + 1) / n_cols) * 55.0
            yield ProgressEvent(
                event_type=EventType.PROGRESS,
                message=f"profiled column {i + 1}/{n_cols}: {col}",
                progress_pct=round(pct, 2),
            )

        correlations: list[dict[str, Any]] = []
        if self.options.mode == "full" and self.options.correlation_method != "none":
            yield ProgressEvent(
                event_type=EventType.PROGRESS,
                message=f"computing {self.options.correlation_method} correlations",
                progress_pct=78.0,
            )
            correlations = compute_correlations(df, self.options)

        dataset_warnings = collect_dataset_warnings(summary, column_profiles)

        profile_result: dict[str, Any] = {
            "summary": summary,
            "columns": column_profiles,
            "correlations": correlations,
            "warnings": dataset_warnings,
        }

        artifact.schema = {c: str(df[c].dtype) for c in df.columns}

        if self.options.mode == "full":
            out = context.artifact_store.path / "artifacts" / artifact.run_id / "profile.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(profile_result, indent=2, default=str))
            artifact.materialized = True
            artifact.path = str(out.relative_to(context.artifact_store.path))
            yield ProgressEvent(
                event_type=EventType.PROGRESS,
                message=f"materialized to {artifact.path}",
                progress_pct=95.0,
                payload={"path": artifact.path},
            )
        else:
            artifact.options = {**artifact.options, "profile_summary": {
                "summary": summary,
                "warnings": dataset_warnings,
            }}

        context.artifact_store.save(artifact)
        context.artifact = artifact

        yield ProgressEvent(
            event_type=EventType.DONE,
            message="profile complete",
            progress_pct=100.0,
            payload={
                "run_id": artifact.run_id,
                "ref": artifact.ref,
                "n_warnings": len(dataset_warnings),
                "warnings": [w for w in dataset_warnings if w["severity"] == "error"],
            },
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "pipeline": "profile",
            "version": "2.0",
            "parent_run_id": self.parent_run_id,
            "options": {
                "mode": self.options.mode,
                "chunk_size": self.options.chunk_size,
                "sample_strategy": self.options.sample_strategy,
                "sample_size": self.options.sample_size,
                "correlation_method": self.options.correlation_method,
                "correlation_threshold": self.options.correlation_threshold,
                "histogram_bin_count": self.options.histogram_bin_count,
                "detect_patterns": self.options.detect_patterns,
            },
            "schema": {"input": "source_config | dataset", "output": "profile"},
            "capabilities": ["chunked", "sample", "materialize", "correlations", "pattern_detection"],
        }