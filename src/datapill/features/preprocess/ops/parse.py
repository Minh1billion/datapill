from typing import Any

import polars as pl


def trim(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    return df.with_columns([pl.col(c).str.strip_chars() for c in cols])


def lower(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    return df.with_columns([pl.col(c).str.to_lowercase() for c in cols])


def upper(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    return df.with_columns([pl.col(c).str.to_uppercase() for c in cols])


def regex_extract(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    col = op["col"]
    pattern = op["pattern"]
    group = op.get("group", 1)
    out_col = op.get("out_col") or f"{col}_extracted"
    return df.with_columns(
        pl.col(col).str.extract(pattern, group_index=group).alias(out_col)
    )


def split_col(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    col = op["col"]
    sep = op["sep"]
    out_cols = op.get("out_cols")
    n = op.get("n")
    result = df.with_columns(
        pl.col(col).str.splitn(sep, n=n or 2).alias("_split_struct")
    )
    if out_cols:
        exprs = [
            pl.col("_split_struct").struct.field(f"field_{i}").alias(name)
            for i, name in enumerate(out_cols)
        ]
    else:
        n_out = n or 2
        exprs = [
            pl.col("_split_struct").struct.field(f"field_{i}").alias(f"{col}_{i}")
            for i in range(n_out)
        ]
    return result.with_columns(exprs).drop("_split_struct")


def parse_datetime(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    fmt = op.get("format")
    return df.with_columns([
        pl.col(c).str.to_datetime(format=fmt, strict=False) for c in cols
    ])


def extract_datetime_part(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    col = op["col"]
    parts: list[str] = op.get("parts") or ["year", "month", "day"]
    exprs = []
    for part in parts:
        out_col = op.get("out_col") or f"{col}_{part}"
        if part == "year":
            exprs.append(pl.col(col).dt.year().alias(out_col))
        elif part == "month":
            exprs.append(pl.col(col).dt.month().alias(out_col))
        elif part == "day":
            exprs.append(pl.col(col).dt.day().alias(out_col))
        elif part == "hour":
            exprs.append(pl.col(col).dt.hour().alias(out_col))
        elif part == "minute":
            exprs.append(pl.col(col).dt.minute().alias(out_col))
        elif part == "dow":
            exprs.append(pl.col(col).dt.weekday().alias(out_col))
        elif part == "week":
            exprs.append(pl.col(col).dt.week().alias(out_col))
        elif part == "quarter":
            exprs.append(pl.col(col).dt.quarter().alias(out_col))
    return df.with_columns(exprs)


def date_diff(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    col_a = op["col_a"]
    col_b = op["col_b"]
    unit = op.get("unit", "day")
    out_col = op.get("out_col") or f"{col_a}_{col_b}_diff_{unit}"
    return df.with_columns(
        (pl.col(col_a) - pl.col(col_b)).dt.total_days().alias(out_col)
        if unit == "day"
        else (pl.col(col_a) - pl.col(col_b)).dt.total_hours().alias(out_col)
        if unit == "hour"
        else (pl.col(col_a) - pl.col(col_b)).dt.total_minutes().alias(out_col)
    )