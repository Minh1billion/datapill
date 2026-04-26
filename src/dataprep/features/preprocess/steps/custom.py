import polars as pl
from RestrictedPython import compile_restricted, safe_globals

from ..schema import StepConfig, StepResult
from .base import BaseStep


class CustomPython(BaseStep):
    def apply(self, df: pl.DataFrame) -> tuple[pl.DataFrame, StepResult]:
        code: str = self.config.params.get("code", "")
        func_name: str = self.config.params.get("func", "transform")
        before = df

        byte_code = compile_restricted(code, filename="<custom_step>", mode="exec")

        glb = {**safe_globals, "pl": pl, "__builtins__": safe_globals["__builtins__"]}
        exec(byte_code, glb)

        func = glb.get(func_name)
        if func is None:
            raise ValueError(f"Function '{func_name}' not found in custom code")

        result = func(df)
        if not isinstance(result, pl.DataFrame):
            raise TypeError(f"Custom function must return pl.DataFrame, got {type(result)}")

        cols = self.config.columns or df.columns
        return result, StepResult(
            step=self.config.step,
            columns_affected=cols,
            row_delta=len(result) - len(before),
            null_delta=self._null_delta(before, result, [c for c in cols if c in result.columns]),
            dtype_changes=self._dtype_changes(before, result, [c for c in cols if c in result.columns]),
        )