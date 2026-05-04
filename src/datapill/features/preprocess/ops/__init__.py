from typing import Any

import polars as pl

from . import clean, compose, parse, reshape, schema, transform

_DISPATCH: dict[str, dict[str, Any]] = {
    "schema": {
        "cast": schema.cast,
        "rename": schema.rename,
        "drop_columns": schema.drop_columns,
        "select_columns": schema.select_columns,
        "reorder_columns": schema.reorder_columns,
    },
    "clean": {
        "fill_null": clean.fill_null,
        "drop_null": clean.drop_null,
        "impute": clean.impute,
        "clip": clean.clip,
        "winsorize": clean.winsorize,
        "drop_outlier": clean.drop_outlier,
        "flag_outlier": clean.flag_outlier,
        "replace_value": clean.replace_value,
    },
    "transform": {
        "normalize": transform.normalize,
        "standardize": transform.standardize,
        "log_transform": transform.log_transform,
        "bin": transform.bin,
        "rank": transform.rank,
        "power_transform": transform.power_transform,
        "encode": transform.encode,
        "math_expr": transform.math_expr,
    },
    "parse": {
        "trim": parse.trim,
        "lower": parse.lower,
        "upper": parse.upper,
        "regex_extract": parse.regex_extract,
        "split_col": parse.split_col,
        "parse_datetime": parse.parse_datetime,
        "extract_datetime_part": parse.extract_datetime_part,
        "date_diff": parse.date_diff,
    },
    "reshape": {
        "filter_rows": reshape.filter_rows,
        "dedup": reshape.dedup,
        "sort": reshape.sort,
        "add_column": reshape.add_column,
        "explode": reshape.explode,
        "pivot": reshape.pivot,
        "unpivot": reshape.unpivot,
    },
    "compose": {
        "window_agg": compose.window_agg,
        "group_agg": compose.group_agg,
        "feature_cross": compose.feature_cross,
        "resample": compose.resample,
        "sample": compose.sample,
        "custom_expr": compose.custom_expr,
    },
}

_REQUIRED: dict[str, dict[str, list[str]]] = {
    "schema": {
        "cast": ["col", "dtype"],
        "rename": ["mapping"],
        "drop_columns": ["cols"],
        "select_columns": ["cols"],
        "reorder_columns": ["cols"],
    },
    "clean": {
        "fill_null": ["value"],
        "drop_null": [],
        "impute": [],
        "clip": [],
        "winsorize": [],
        "drop_outlier": [],
        "flag_outlier": [],
        "replace_value": ["mapping"],
    },
    "transform": {
        "normalize": [],
        "standardize": [],
        "log_transform": [],
        "bin": ["col", "breaks"],
        "rank": [],
        "power_transform": [],
        "encode": ["col"],
        "math_expr": ["out_col", "expr"],
    },
    "parse": {
        "trim": [],
        "lower": [],
        "upper": [],
        "regex_extract": ["col", "pattern"],
        "split_col": ["col", "sep"],
        "parse_datetime": [],
        "extract_datetime_part": ["col"],
        "date_diff": ["col_a", "col_b"],
    },
    "reshape": {
        "filter_rows": ["expr"],
        "dedup": [],
        "sort": [],
        "add_column": ["col", "value"],
        "explode": ["col"],
        "pivot": ["on", "index", "values"],
        "unpivot": [],
    },
    "compose": {
        "window_agg": ["col"],
        "group_agg": ["by", "aggs"],
        "feature_cross": ["col_a", "col_b"],
        "resample": ["time_col", "every", "agg_col"],
        "sample": [],
        "custom_expr": ["out_col", "expr"],
    },
}


def apply_op(df: pl.DataFrame, op: dict[str, Any]) -> pl.DataFrame:
    group = op.get("group")
    type_ = op.get("type")
    if group not in _DISPATCH:
        raise ValueError(f"unknown op group: {group!r}")
    if type_ not in _DISPATCH[group]:
        raise ValueError(f"unknown op type: {type_!r} in group {group!r}")
    return _DISPATCH[group][type_](df, op)


def validate_op(op: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    group = op.get("group")
    type_ = op.get("type")
    if not group:
        errors.append("op missing 'group'")
        return errors
    if not type_:
        errors.append("op missing 'type'")
        return errors
    if group not in _DISPATCH:
        errors.append(f"unknown group: {group!r}")
        return errors
    if type_ not in _DISPATCH[group]:
        errors.append(f"unknown type: {type_!r} in group {group!r}")
        return errors
    for key in _REQUIRED[group][type_]:
        if key not in op:
            errors.append(f"{group}.{type_} missing required param: {key!r}")
    return errors