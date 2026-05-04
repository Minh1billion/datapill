from typing import Any

import polars as pl


def normalize(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    exprs = []
    for c in cols:
        mn = df[c].min()
        mx = df[c].max()
        denom = mx - mn
        if denom == 0:
            exprs.append(pl.lit(0.0).alias(c))
        else:
            exprs.append(((pl.col(c) - mn) / denom).alias(c))
    return df.with_columns(exprs)


def standardize(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    fit_params = op.get("fit_params", {})
    exprs = []
    computed: dict[str, dict] = {}
    for c in cols:
        mean = fit_params.get(c, {}).get("mean") if fit_params else None
        std = fit_params.get(c, {}).get("std") if fit_params else None
        if mean is None:
            mean = df[c].mean()
        if std is None:
            std = df[c].std()
        computed[c] = {"mean": mean, "std": std}
        if std == 0:
            exprs.append(pl.lit(0.0).alias(c))
        else:
            exprs.append(((pl.col(c) - mean) / std).alias(c))
    op["fit_params"] = computed
    return df.with_columns(exprs)


def log_transform(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    base = op.get("base", "natural")
    exprs = []
    for c in cols:
        if base == "natural":
            exprs.append(pl.col(c).log(base=2.718281828).alias(c))
        elif base == "log10":
            exprs.append(pl.col(c).log(base=10).alias(c))
        elif base == "log2":
            exprs.append(pl.col(c).log(base=2).alias(c))
    return df.with_columns(exprs)


def bin(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    breaks = op["breaks"]
    col = op["col"]
    labels = op.get("labels")
    out_col = op.get("out_col") or f"{col}_bin"
    return df.with_columns(
        pl.col(col).cut(breaks, labels=labels).alias(out_col)
    )


def rank(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    method = op.get("method", "average")
    return df.with_columns([
        pl.col(c).rank(method=method).alias(f"{c}_rank") for c in cols
    ])


def power_transform(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    cols = op.get("cols") or df.columns
    method = op.get("method", "yeo-johnson")
    exprs = []
    for c in cols:
        if method == "sqrt":
            exprs.append(pl.col(c).sqrt().alias(c))
        elif method == "square":
            exprs.append((pl.col(c) ** 2).alias(c))
        elif method == "cbrt":
            exprs.append((pl.col(c).abs() ** (1 / 3) * pl.col(c).sign()).alias(c))
    return df.with_columns(exprs)


def encode(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    col = op["col"]
    method = op.get("method", "onehot")
    fit_params = op.get("fit_params", {})

    if method == "label":
        categories = fit_params.get("categories") or sorted(df[col].drop_nulls().unique().to_list())
        mapping = {v: i for i, v in enumerate(categories)}
        op["fit_params"] = {"categories": categories}
        return df.with_columns(pl.col(col).replace(mapping).cast(pl.Int32))

    if method == "onehot":
        categories = fit_params.get("categories") or sorted(df[col].drop_nulls().unique().to_list())
        op["fit_params"] = {"categories": categories}
        exprs = [
            (pl.col(col) == cat).cast(pl.Int8).alias(f"{col}_{cat}")
            for cat in categories
        ]
        return df.with_columns(exprs).drop(col)

    if method == "ordinal":
        mapping: dict = op["mapping"]
        op["fit_params"] = {"mapping": mapping}
        return df.with_columns(pl.col(col).replace(mapping).cast(pl.Int32))

    raise ValueError(f"unknown encode method: {method!r}")


def math_expr(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    out_col = op["out_col"]
    expr_str: str = op["expr"]
    return df.with_columns(pl.eval(expr_str).alias(out_col))