import polars as pl

from ..schema import StepConfig, StepResult
from .base import BaseStep


class ClipIQR(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns or [c for c in df.columns if df[c].dtype.is_numeric()]
        before = df
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            df = df.with_columns(pl.col(col).clip(lo, hi))
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta=self._null_delta(before, df, cols),
            dtype_changes=self._dtype_changes(before, df, cols),
        )


class ClipZScore(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns or [c for c in df.columns if df[c].dtype.is_numeric()]
        threshold = self.config.params.get("threshold", 3.0)
        before = df
        for col in cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                continue
            lo = mean - threshold * std
            hi = mean + threshold * std
            df = df.with_columns(pl.col(col).clip(lo, hi))
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta=self._null_delta(before, df, cols),
            dtype_changes=self._dtype_changes(before, df, cols),
        )