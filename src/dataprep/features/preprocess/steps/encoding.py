import polars as pl

from ..schema import StepConfig, StepResult
from .base import BaseStep


class OneHotEncoder(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns or df.columns
        before = df
        for col in cols:
            categories = df[col].drop_nulls().unique().sort().to_list()
            for cat in categories:
                df = df.with_columns((pl.col(col) == cat).cast(pl.Int8).alias(f"{col}__{cat}"))
            df = df.drop(col)
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta={},
            dtype_changes={col: (str(before[col].dtype), "Int8 (one-hot)") for col in cols},
        )


class OrdinalEncoder(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        cols = self.config.columns or df.columns
        before = df
        for col in cols:
            order: list | None = self.config.params.get("order", {}).get(col)
            if order:
                mapping = {v: i for i, v in enumerate(order)}
            else:
                categories = df[col].drop_nulls().unique().sort().to_list()
                mapping = {v: i for i, v in enumerate(categories)}

            df = df.with_columns(
                pl.col(col)
                .map_elements(lambda v: mapping.get(v), return_dtype=pl.Int32)
                .alias(col)
            )
        return df, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=0,
            null_delta=self._null_delta(before, df, cols),
            dtype_changes=self._dtype_changes(before, df, cols),
        )