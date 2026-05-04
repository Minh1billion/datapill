from typing import Any

import polars as pl


def fill_null(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    return df.with_columns([pl.col(c).fill_null(op["value"]) for c in cols])


def drop_null(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    return df.drop_nulls(subset=cols)


def impute(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    strategy = op.get("strategy", "mean")
    exprs = []
    for c in cols:
        if strategy == "mean":
            fill = df[c].mean()
        elif strategy == "median":
            fill = df[c].median()
        elif strategy == "mode":
            mode = df[c].drop_nulls().mode()
            fill = mode[0] if len(mode) else None
        else:
            fill = op.get("value")
        exprs.append(pl.col(c).fill_null(fill))
    return df.with_columns(exprs)


def clip(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    return df.with_columns([
        pl.col(c).clip(op.get("min"), op.get("max")) for c in cols
    ])


def winsorize(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    lower_pct = op.get("lower", 0.05)
    upper_pct = op.get("upper", 0.95)
    exprs = []
    for c in cols:
        lower = df[c].quantile(lower_pct, interpolation="nearest")
        upper = df[c].quantile(upper_pct, interpolation="nearest")
        exprs.append(pl.col(c).clip(lower, upper))
    return df.with_columns(exprs)


def drop_outlier(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    method = op.get("method", "iqr")
    mask = pl.lit(True)
    for c in cols:
        if method == "iqr":
            q1 = df[c].quantile(0.25, interpolation="nearest")
            q3 = df[c].quantile(0.75, interpolation="nearest")
            iqr = q3 - q1
            mask = mask & pl.col(c).is_between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        elif method == "zscore":
            threshold = op.get("threshold", 3.0)
            mean = df[c].mean()
            std = df[c].std()
            mask = mask & ((pl.col(c) - mean).abs() / std <= threshold)
    return df.filter(mask)


def flag_outlier(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    method = op.get("method", "iqr")
    exprs = []
    for c in cols:
        flag_col = op.get("flag_col") or f"{c}_is_outlier"
        if method == "iqr":
            q1 = df[c].quantile(0.25, interpolation="nearest")
            q3 = df[c].quantile(0.75, interpolation="nearest")
            iqr = q3 - q1
            exprs.append(
                (~pl.col(c).is_between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)).alias(flag_col)
            )
        elif method == "zscore":
            threshold = op.get("threshold", 3.0)
            mean = df[c].mean()
            std = df[c].std()
            exprs.append(
                ((pl.col(c) - mean).abs() / std > threshold).alias(flag_col)
            )
    return df.with_columns(exprs)


def replace_value(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    mapping: dict = op["mapping"]
    return df.with_columns([
        pl.col(c).replace(mapping) for c in cols
    ])