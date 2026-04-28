import polars as pl

from ..schema import StepConfig, StepResult
from .base import BaseStep


class StandardScaler(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns or [c for c in df.columns if df[c].dtype.is_numeric()]
        before = df
        for col in cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                continue
            df = df.with_columns(((pl.col(col) - mean) / std).alias(col))
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta=self._null_delta(before, df, cols),
            dtype_changes=self._dtype_changes(before, df, cols),
        )


class MinMaxScaler(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns or [c for c in df.columns if df[c].dtype.is_numeric()]
        before = df
        for col in cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val == min_val:
                continue
            df = df.with_columns(((pl.col(col) - min_val) / (max_val - min_val)).alias(col))
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta=self._null_delta(before, df, cols),
            dtype_changes=self._dtype_changes(before, df, cols),
        )


class RobustScaler(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns or [c for c in df.columns if df[c].dtype.is_numeric()]
        before = df
        for col in cols:
            median = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            df = df.with_columns(((pl.col(col) - median) / iqr).alias(col))
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta=self._null_delta(before, df, cols),
            dtype_changes=self._dtype_changes(before, df, cols),
        )