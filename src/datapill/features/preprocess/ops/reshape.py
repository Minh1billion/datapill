from typing import Any

import polars as pl


def filter_rows(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    return df.filter(pl.Expr.deserialize(op["expr"].encode(), format="json"))


def dedup(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols")
    keep = op.get("keep", "first")
    return df.unique(subset=cols, keep=keep, maintain_order=True)


def sort(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    descending = op.get("descending", False)
    if isinstance(descending, bool):
        descending = [descending] * len(cols)
    return df.sort(cols, descending=descending)


def add_column(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    return df.with_columns(pl.lit(op["value"]).alias(op["col"]))


def explode(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    return df.explode(op["col"])


def pivot(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    return df.pivot(
        on=op["on"],
        index=op["index"],
        values=op["values"],
        aggregate_function=op.get("aggregate_function", "first"),
    )


def unpivot(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    return df.unpivot(
        on=op.get("on"),
        index=op.get("index"),
        variable_name=op.get("variable_name", "variable"),
        value_name=op.get("value_name", "value"),
    )