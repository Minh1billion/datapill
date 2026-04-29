import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, Any
import polars as pl

from datapill.core.interfaces import FeaturePipeline, ValidationResult, ExecutionPlan
from datapill.core.context import PipelineContext
from datapill.core.events import ProgressEvent, EventType
from .stats import ColumnStatsComputer, compute_correlation_matrix
from .schema import (
    ProfileDetail, ProfileSummary, DatasetMeta,
    ColumnProfile, ColumnSummary, CorrelationPair, DatasetWarning,
)

_COLUMN_BATCH = 10
_KAFKA_RESTREAM_WARNING = (
    "Warning: input is a Kafka ref - re-consuming from topic. "
    "Offsets will advance and data may differ from original ingest."
)
_RESTREAM_WARNING = (
    "Warning: input is a ref artifact - re-streaming from source. "
    "Source must remain available."
)


@dataclass
class ProfileOptions:
    mode: str = "full"
    sample_strategy: str = "none"
    sample_size: int = 100_000
    batch_size: int = 50_000
    numeric_percentiles: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    cardinality_limit: int = 50
    text_sample_limit: int = 10
    detect_patterns: bool = True
    correlation_method: str = "pearson"
    correlation_threshold: float = 0.5
    histogram_bin_count: int = 20


class ProfilePipeline(FeaturePipeline):
    def __init__(self, options: ProfileOptions | None = None) -> None:
        self.options = options or ProfileOptions()

    def validate(self, context: PipelineContext) -> ValidationResult:
        errors: list[str] = []
        if self.options.mode not in ("full", "summary"):
            errors.append(f"Invalid mode: {self.options.mode}")
        if self.options.sample_strategy not in ("none", "random", "reservoir"):
            errors.append(f"Invalid sample_strategy: {self.options.sample_strategy}")
        return ValidationResult(ok=len(errors) == 0, errors=errors)

    def plan(self, context: PipelineContext) -> ExecutionPlan:
        steps = ["load_data", "sampling", "compute_stats", "compute_correlations", "build_output"]
        return ExecutionPlan(steps=[{"name": s} for s in steps])

    async def execute(
        self, plan: ExecutionPlan, context: PipelineContext
    ) -> AsyncGenerator[ProgressEvent, None]:
        profile_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()

        yield ProgressEvent(EventType.STARTED, "Profile started", progress_pct=0.0)

        input_ref = plan.metadata.get("input_artifact_id")
        df: pl.DataFrame | None = None

        if input_ref and context.artifact_store.is_ref(input_ref):
            ref = await context.artifact_store.load_ref(input_ref)
            from datapill.cli._shared import rebuild_connector_from_ref
            connector, query = rebuild_connector_from_ref(ref)

            if "Kafka" in ref.get("source_type", ""):
                yield ProgressEvent(EventType.PROGRESS, _KAFKA_RESTREAM_WARNING, progress_pct=0.02)
            else:
                yield ProgressEvent(EventType.PROGRESS, _RESTREAM_WARNING, progress_pct=0.02)

            chunks: list[pl.DataFrame] = []
            async for chunk in connector.read_stream(query, ref.get("options", {})):
                if not chunk.is_empty():
                    chunks.append(chunk)
            df = pl.concat(chunks) if chunks else pl.DataFrame()
            await connector.close()

        elif input_ref:
            if self.options.sample_strategy == "none":
                df = context.artifact_store.scan_parquet(input_ref).collect()
            else:
                lf = context.artifact_store.scan_parquet(input_ref)
                df = self._apply_sampling_lazy(lf)
        elif "dataframe" in plan.metadata:
            df = plan.metadata["dataframe"]
            df = self._apply_sampling_eager(df)
        else:
            yield ProgressEvent(EventType.ERROR, "No input data provided")
            return

        if self.options.sample_strategy != "none" and input_ref and context.artifact_store.is_ref(input_ref):
            df = self._apply_sampling_eager(df)

        yield ProgressEvent(EventType.PROGRESS, "Sampling complete", progress_pct=0.1,
                            payload={"rows": len(df), "cols": len(df.columns)})

        stats_computer = ColumnStatsComputer(
            cardinality_limit=self.options.cardinality_limit,
            text_sample_limit=self.options.text_sample_limit,
            detect_patterns=self.options.detect_patterns,
            numeric_percentiles=self.options.numeric_percentiles,
            histogram_bin_count=self.options.histogram_bin_count,
        )

        column_profiles: list[dict] = []
        total_cols = len(df.columns)

        for batch_start in range(0, total_cols, _COLUMN_BATCH):
            batch_cols = df.columns[batch_start:batch_start + _COLUMN_BATCH]
            for col in batch_cols:
                try:
                    col_stats = stats_computer.compute(df[col])
                    column_profiles.append(col_stats)
                except Exception as exc:
                    column_profiles.append({
                        "name": col,
                        "dtype_physical": str(df[col].dtype),
                        "dtype_inferred": "unknown",
                        "null_count": 0,
                        "null_pct": 0.0,
                        "distinct_count": 0,
                        "distinct_pct": 0.0,
                        "is_unique": False,
                        "warnings": [f"PROFILE_ERROR: {exc}"],
                    })

            done_cols = min(batch_start + _COLUMN_BATCH, total_cols)
            pct = 0.1 + (done_cols / total_cols) * 0.65
            yield ProgressEvent(
                EventType.PROGRESS,
                f"Profiling columns {done_cols}/{total_cols}",
                progress_pct=round(pct, 3),
            )

        yield ProgressEvent(EventType.PROGRESS, "Computing correlations", progress_pct=0.80)
        correlations: list[dict] = []
        if self.options.correlation_method != "none":
            correlations = compute_correlation_matrix(
                df,
                method=self.options.correlation_method,
                threshold=self.options.correlation_threshold,
            )

        yield ProgressEvent(EventType.PROGRESS, "Building output", progress_pct=0.92)

        detail = self._build_detail(profile_id, df, column_profiles, correlations)
        summary = self._build_summary(detail)

        detail_id = f"{profile_id}_detail"
        summary_id = f"{profile_id}_summary"
        await context.artifact_store.save_json(detail_id, detail.model_dump(), feature="profile")
        await context.artifact_store.save_json(summary_id, summary.model_dump(), feature="profile")

        duration = time.perf_counter() - t0
        yield ProgressEvent(
            EventType.DONE,
            f"Profile complete in {duration:.1f}s - {total_cols} columns, {len(df):,} rows",
            progress_pct=1.0,
            payload={
                "profile_id": profile_id,
                "detail_artifact_id": detail_id,
                "summary_artifact_id": summary_id,
                "duration_s": round(duration, 3),
            },
        )

    def _apply_sampling_lazy(self, lf: pl.LazyFrame) -> pl.DataFrame:
        df = lf.collect()
        if self.options.sample_strategy == "none" or len(df) <= self.options.sample_size:
            return df
        if self.options.sample_strategy == "random":
            return df.sample(n=self.options.sample_size, seed=42)
        if self.options.sample_strategy == "reservoir":
            return _reservoir_sample(df, self.options.sample_size)
        return df

    def _apply_sampling_eager(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.options.sample_strategy == "none" or len(df) <= self.options.sample_size:
            return df
        if self.options.sample_strategy == "random":
            return df.sample(n=self.options.sample_size, seed=42)
        if self.options.sample_strategy == "reservoir":
            return _reservoir_sample(df, self.options.sample_size)
        return df

    def _build_detail(
        self,
        profile_id: str,
        df: pl.DataFrame,
        column_stats: list[dict],
        correlations: list[dict],
    ) -> ProfileDetail:
        n = len(df)
        schema_hash = _schema_hash(df)
        sampled = self.options.sample_strategy != "none"

        dataset_warnings: list[DatasetWarning] = []
        for col_stat in column_stats:
            for w in col_stat.get("warnings", []):
                if w.startswith("HIGH_NULL") or w.startswith("CONSTANT"):
                    dataset_warnings.append(DatasetWarning(
                        type=w,
                        column=col_stat["name"],
                        message=f"Column '{col_stat['name']}': {w}",
                    ))

        return ProfileDetail(
            profile_id=profile_id,
            dataset=DatasetMeta(
                row_count=n,
                column_count=len(df.columns),
                memory_mb=round(df.estimated_size("mb"), 3),
                sampled=sampled,
                sample_strategy=self.options.sample_strategy,
                sample_size=n if sampled else None,
                schema_hash=schema_hash,
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
            columns=[ColumnProfile(**_pad_col_profile(cs)) for cs in column_stats],
            correlations=[CorrelationPair(**c) for c in correlations],
            warnings=dataset_warnings,
        )

    def _build_summary(self, detail: ProfileDetail) -> ProfileSummary:
        high_level: list[str] = []
        seen: set[str] = set()
        for w in detail.warnings:
            if w.type not in seen:
                high_level.append(w.message)
                seen.add(w.type)

        col_summaries: list[ColumnSummary] = []
        for col in detail.columns:
            top3 = (
                [v.value for v in col.top_values[:3]]
                if col.top_values else None
            )
            col_summaries.append(ColumnSummary(
                name=col.name,
                dtype=col.dtype_inferred,
                null_pct=col.null_pct,
                distinct_count=col.distinct_count,
                min=col.min,
                max=col.max,
                mean=col.mean,
                top_3_values=top3,
                warnings=col.warnings,
            ))

        return ProfileSummary(
            profile_id=detail.profile_id,
            dataset={
                "row_count": detail.dataset.row_count,
                "column_count": detail.dataset.column_count,
                "sampled": detail.dataset.sampled,
                "sample_size": detail.dataset.sample_size,
            },
            columns=col_summaries,
            high_level_warnings=high_level,
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "feature": "profile",
            "version": "1.0",
            "options": {
                "mode": self.options.mode,
                "sample_strategy": self.options.sample_strategy,
                "sample_size": self.options.sample_size,
                "correlation_method": self.options.correlation_method,
            },
        }


def _schema_hash(df: pl.DataFrame) -> str:
    schema_str = json.dumps(
        {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        sort_keys=True,
    )
    return hashlib.md5(schema_str.encode()).hexdigest()[:12]


def _reservoir_sample(df: pl.DataFrame, k: int) -> pl.DataFrame:
    import random
    n = len(df)
    if n <= k:
        return df
    indices = list(range(k))
    for i in range(k, n):
        j = random.randint(0, i)
        if j < k:
            indices[j] = i
    return df[sorted(indices)]


_COL_PROFILE_DEFAULTS = {
    "min": None, "max": None, "mean": None, "median": None,
    "std": None, "variance": None, "skewness": None, "kurtosis": None,
    "percentiles": None, "histogram": None, "top_values": None,
    "pattern_matches": None, "sample_values": None,
}


def _pad_col_profile(cs: dict) -> dict:
    merged = {**_COL_PROFILE_DEFAULTS, **cs}
    if "warnings" not in merged:
        merged["warnings"] = []
    return merged