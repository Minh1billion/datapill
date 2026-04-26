import uuid
from pathlib import Path

import polars as pl

from .schema import RunReport, StepConfig, StepResult
from .steps import build_step

_DRY_RUN_ROWS = 1000
_PREVIEW_ROWS = 10
_CHECKPOINT_DIR = Path("artifacts")

_NUMERIC_STEPS = {
    "impute_mean", "impute_median", "clip_iqr", "clip_zscore",
    "standard_scaler", "minmax_scaler", "robust_scaler",
}


class PreprocessPipeline:
    def __init__(self, steps: list[StepConfig], checkpoint: bool = False) -> None:
        self.steps = steps
        self.checkpoint = checkpoint
        self.run_id = uuid.uuid4().hex[:8]

    def run(self, df: pl.DataFrame, dry_run: bool = False) -> RunReport:
        warnings = self._detect_conflicts()
        source = df.head(_DRY_RUN_ROWS) if dry_run else df

        results: list[StepResult] = []
        current = source

        for n, cfg in enumerate(self.steps):
            step = build_step(cfg)
            self._validate_numeric_requirement(cfg, current)
            current, result = step.apply(current)
            results.append(result)

            if self.checkpoint and not dry_run:
                self._save_checkpoint(current, n)

        report = RunReport(
            run_id=self.run_id,
            dry_run=dry_run,
            steps=results,
            warnings=warnings,
        )

        if dry_run:
            report.preview_rows = current.head(_PREVIEW_ROWS).to_dicts()
            report.output_schema = {col: str(dtype) for col, dtype in zip(current.columns, current.dtypes)}

        return report

    def resume(self, df: pl.DataFrame, from_step: int) -> RunReport:
        checkpoint_path = _CHECKPOINT_DIR / f"{self.run_id}_checkpoint_step_{from_step - 1}.parquet"
        if checkpoint_path.exists():
            df = pl.read_parquet(checkpoint_path)

        partial_pipeline = PreprocessPipeline(
            steps=self.steps[from_step:],
            checkpoint=self.checkpoint,
        )
        partial_pipeline.run_id = self.run_id
        return partial_pipeline.run(df)

    def _save_checkpoint(self, df: pl.DataFrame, step_index: int) -> None:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = _CHECKPOINT_DIR / f"{self.run_id}_checkpoint_step_{step_index}.parquet"
        df.write_parquet(path)

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