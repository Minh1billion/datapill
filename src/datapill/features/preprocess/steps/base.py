from abc import ABC, abstractmethod

import polars as pl

from ..schema import StepConfig, StepResult


class BaseStep(ABC):
    def __init__(self, config: StepConfig) -> None:
        self.config = config

    @abstractmethod
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        ...

    def _null_delta(self, before: pl.DataFrame, after: pl.DataFrame, columns: list[str]) -> dict[str, int]:
        result = {}
        for col in columns:
            if col in before.columns and col in after.columns:
                before_nulls = before[col].null_count()
                after_nulls = after[col].null_count()
                if before_nulls != after_nulls:
                    result[col] = after_nulls - before_nulls
        return result

    def _dtype_changes(self, before: pl.DataFrame, after: pl.DataFrame, columns: list[str]) -> dict[str, tuple[str, str]]:
        result = {}
        for col in columns:
            if col in before.columns and col in after.columns:
                b = str(before[col].dtype)
                a = str(after[col].dtype)
                if b != a:
                    result[col] = (b, a)
        return result