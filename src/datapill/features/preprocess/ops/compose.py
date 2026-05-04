from typing import Any

import polars as pl


def window_agg(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    col = op["col"]
    fn = op.get("fn", "rolling_mean")
    window = op.get("window", 7)
    partition_by = op.get("partition_by")
    out_col = op.get("out_col") or f"{col}_{fn}_{window}"

    if fn == "rolling_mean":
        expr = pl.col(col).rolling_mean(window_size=window)
    elif fn == "rolling_sum":
        expr = pl.col(col).rolling_sum(window_size=window)
    elif fn == "rolling_std":
        expr = pl.col(col).rolling_std(window_size=window)
    elif fn == "rolling_min":
        expr = pl.col(col).rolling_min(window_size=window)
    elif fn == "rolling_max":
        expr = pl.col(col).rolling_max(window_size=window)
    elif fn == "lag":
        expr = pl.col(col).shift(op.get("n", 1))
    elif fn == "lead":
        expr = pl.col(col).shift(-op.get("n", 1))
    else:
        raise ValueError(f"unknown window fn: {fn!r}")

    if partition_by:
        return df.with_columns(expr.over(partition_by).alias(out_col))
    return df.with_columns(expr.alias(out_col))


def group_agg(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    by = op["by"]
    aggs_spec: list[dict] = op["aggs"]
    exprs = []
    for agg in aggs_spec:
        col = agg["col"]
        fn = agg["fn"]
        out_col = agg.get("out_col") or f"{col}_{fn}"
        if fn == "sum":
            exprs.append(pl.col(col).sum().alias(out_col))
        elif fn == "mean":
            exprs.append(pl.col(col).mean().alias(out_col))
        elif fn == "min":
            exprs.append(pl.col(col).min().alias(out_col))
        elif fn == "max":
            exprs.append(pl.col(col).max().alias(out_col))
        elif fn == "count":
            exprs.append(pl.col(col).count().alias(out_col))
        elif fn == "std":
            exprs.append(pl.col(col).std().alias(out_col))
        elif fn == "median":
            exprs.append(pl.col(col).median().alias(out_col))
        elif fn == "nunique":
            exprs.append(pl.col(col).n_unique().alias(out_col))
    return df.group_by(by).agg(exprs)


def feature_cross(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    col_a = op["col_a"]
    col_b = op["col_b"]
    out_col = op.get("out_col") or f"{col_a}_x_{col_b}"
    sep = op.get("sep", "_")
    return df.with_columns(
        (pl.col(col_a).cast(pl.Utf8) + sep + pl.col(col_b).cast(pl.Utf8)).alias(out_col)
    )


def resample(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    time_col = op["time_col"]
    every = op["every"]
    agg_col = op["agg_col"]
    fn = op.get("fn", "mean")
    out_col = op.get("out_col") or f"{agg_col}_{fn}"
    agg_expr = getattr(pl.col(agg_col), fn)()
    return (
        df.sort(time_col)
        .group_by_dynamic(time_col, every=every)
        .agg(agg_expr.alias(out_col))
    )


def sample(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    n = op.get("n")
    frac = op.get("frac")
    seed = op.get("seed", 42)
    if n is not None:
        return df.sample(n=n, seed=seed)
    if frac is not None:
        return df.sample(fraction=frac, seed=seed)
    raise ValueError("sample op requires either 'n' or 'frac'")


def custom_expr(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    out_col = op["out_col"]
    expr_str: str = op["expr"]
    return df.with_columns(pl.eval(expr_str).alias(out_col))