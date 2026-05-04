from typing import Any

import polars as pl


def cast(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    return df.with_columns(pl.col(op["col"]).cast(getattr(pl, op["dtype"])))


def rename(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    return df.rename(op["mapping"])


def drop_columns(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    return df.drop(op["cols"])


def select_columns(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    return df.select(op["cols"])


def reorder_columns(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    rest = [c for c in df.columns if c not in op["cols"]]
    return df.select(op["cols"] + rest)