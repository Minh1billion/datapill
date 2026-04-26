import polars as pl

from ..schema import StepConfig, StepResult
from .base import BaseStep


class SelectColumns(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns
        before = df
        df = df.select(cols)
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta={},
            dtype_changes={},
        )


class DropColumns(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns
        df = df.drop(cols)
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta={},
            dtype_changes={},
        )


class RenameColumns(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        mapping: dict[str, str] = self.config.params.get("mapping", {})
        df = df.rename(mapping)
        return df, StepResult(
            step=self.config.step,
            columns_affected=list(mapping.keys()),
            row_delta=0,
            null_delta={},
            dtype_changes={},
        )


class CastDtype(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        casts: dict[str, str] = self.config.params.get("casts", {})
        before = df
        dtype_map = {
            "int8": pl.Int8, "int16": pl.Int16, "int32": pl.Int32, "int64": pl.Int64,
            "float32": pl.Float32, "float64": pl.Float64,
            "str": pl.String, "string": pl.String, "utf8": pl.String,
            "bool": pl.Boolean, "date": pl.Date, "datetime": pl.Datetime,
        }
        exprs = []
        for col, dtype_str in casts.items():
            target = dtype_map.get(dtype_str.lower())
            if target is None:
                raise ValueError(f"Unknown dtype: {dtype_str}")
            exprs.append(pl.col(col).cast(target))
        df = df.with_columns(exprs)
        return df, StepResult(
            step=self.config.step,
            columns_affected=list(casts.keys()),
            row_delta=0,
            null_delta=self._null_delta(before, df, list(casts.keys())),
            dtype_changes=self._dtype_changes(before, df, list(casts.keys())),
        )


class Deduplicate(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        subset = self.config.columns or None
        before_len = len(df)
        df = df.unique(subset=subset, keep="first")
        return df, StepResult(
            step=self.config.step,
            columns_affected=subset or df.columns,
            row_delta=len(df) - before_len,
            null_delta={},
            dtype_changes={},
        )