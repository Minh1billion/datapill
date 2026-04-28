import polars as pl

from ..schema import StepConfig, StepResult
from .base import BaseStep


class ImputeMean(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns or [c for c in df.columns if df[c].dtype.is_numeric()]
        before = df
        for col in cols:
            mean = df[col].mean()
            df = df.with_columns(pl.col(col).fill_null(mean))
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta=self._null_delta(before, df, cols),
            dtype_changes=self._dtype_changes(before, df, cols),
        )


class ImputeMedian(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns or [c for c in df.columns if df[c].dtype.is_numeric()]
        before = df
        for col in cols:
            median = df[col].median()
            df = df.with_columns(pl.col(col).fill_null(median))
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta=self._null_delta(before, df, cols),
            dtype_changes=self._dtype_changes(before, df, cols),
        )


class ImputeMode(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns or df.columns
        before = df
        for col in cols:
            mode_series = df[col].drop_nulls().mode()
            if len(mode_series) == 0:
                continue
            mode_val = mode_series[0]
            df = df.with_columns(pl.col(col).fill_null(mode_val))
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta=self._null_delta(before, df, cols),
            dtype_changes=self._dtype_changes(before, df, cols),
        )


class DropMissing(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns or df.columns
        before_len = len(df)
        df = df.drop_nulls(subset=cols)
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=len(df) - before_len,
            null_delta={},
            dtype_changes={},
        )