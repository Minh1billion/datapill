import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

import polars as pl


@dataclass
class ProfileOptions:
    mode: str = "full"
    chunk_size: int = 100_000
    sample_size: int = 100_000
    sample_strategy: str = "none"
    numeric_percentiles: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    histogram_bin_count: int = 20
    cardinality_limit: int = 100
    pattern_sample_size: int = 5_000
    correlation_method: str = "pearson"
    correlation_threshold: float = 0.3
    detect_patterns: bool = True


_NUMERIC_DTYPES = frozenset({
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
})

_DATETIME_DTYPES = frozenset({pl.Date, pl.Datetime, pl.Duration, pl.Time})

_DTYPE_LABEL: dict[str, str] = {
    "Int8": "integer", "Int16": "integer", "Int32": "integer", "Int64": "integer",
    "UInt8": "integer", "UInt16": "integer", "UInt32": "integer", "UInt64": "integer",
    "Float32": "float", "Float64": "float",
    "Boolean": "boolean",
    "Utf8": "string", "String": "string", "Categorical": "categorical",
    "Date": "datetime", "Datetime": "datetime", "Time": "datetime", "Duration": "datetime",
    "List": "list", "Array": "array", "Struct": "struct", "Binary": "binary",
}

_PATTERNS: dict[str, re.Pattern] = {
    "email":    re.compile(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$"),
    "url":      re.compile(r"^https?://\S+$"),
    "phone_vn": re.compile(r"^(0|\+84)[3-9]\d{8}$"),
    "date_iso": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "uuid":     re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
    "integer_str": re.compile(r"^-?\d+$"),
    "float_str":   re.compile(r"^-?\d+\.\d+$"),
    "ipv4":     re.compile(r"^\d{1,3}(\.\d{1,3}){3}$"),
}

WARNING_HIGH_NULL       = "HIGH_NULL_RATE"
WARNING_CONSTANT        = "CONSTANT_COLUMN"
WARNING_IDENTIFIER      = "POTENTIAL_IDENTIFIER"
WARNING_HIGH_CARDINALITY = "HIGH_CARDINALITY"
WARNING_SKEWED          = "SKEWED_DISTRIBUTION"
WARNING_ALL_ZERO        = "ALL_ZEROS"
WARNING_NEGATIVE        = "HAS_NEGATIVES"
WARNING_HIGH_DUPLICATE  = "HIGH_DUPLICATE_RATE"


def _infer_dtype_label(dtype: pl.DataType) -> str:
    if isinstance(dtype, pl.Decimal):
        return "float"
    if isinstance(dtype, (pl.Datetime, pl.Date, pl.Duration, pl.Time)):
        return "datetime"
    return _DTYPE_LABEL.get(str(dtype), "unknown")


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype in _NUMERIC_DTYPES or isinstance(dtype, pl.Decimal)


def _is_datetime(dtype: pl.DataType) -> bool:
    return isinstance(dtype, (pl.Datetime, pl.Date, pl.Duration, pl.Time))


def _chunk_iter(df: pl.DataFrame, chunk_size: int) -> Iterator[pl.DataFrame]:
    for start in range(0, len(df), chunk_size):
        yield df.slice(start, chunk_size)


def compute_column_profile(
    df: pl.DataFrame,
    col: str,
    options: ProfileOptions,
) -> dict[str, Any]:
    dtype = df[col].dtype
    dtype_label = _infer_dtype_label(dtype)
    n_total = len(df)

    null_count = 0
    distinct_seen: set = set()
    n_rows = 0

    for chunk in _chunk_iter(df, options.chunk_size):
        s = chunk[col]
        null_count += s.null_count()
        n_rows += len(s)
        if len(distinct_seen) < options.cardinality_limit + 1:
            distinct_seen.update(s.drop_nulls().to_list())

    exact_distinct = df[col].n_unique()
    null_pct = round(null_count / n_rows, 6) if n_rows > 0 else 0.0
    is_unique = exact_distinct == n_rows and n_rows > 0

    result: dict[str, Any] = {
        "name": col,
        "dtype_physical": str(dtype),
        "dtype_inferred": dtype_label,
        "null_count": null_count,
        "null_pct": null_pct,
        "distinct_count": exact_distinct,
        "distinct_pct": round(exact_distinct / n_rows, 6) if n_rows > 0 else 0.0,
        "is_unique": is_unique,
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "std": None,
        "variance": None,
        "skewness": None,
        "kurtosis": None,
        "percentiles": None,
        "histogram": None,
        "top_values": None,
        "pattern_matches": None,
        "sample_values": df[col].drop_nulls().head(10).cast(pl.String).to_list(),
        "warnings": [],
    }

    if _is_numeric(dtype) and (n_rows - null_count) > 0:
        cast_expr = pl.col(col).cast(pl.Float64) if isinstance(dtype, pl.Decimal) else pl.col(col)

        col_min = df.select(cast_expr.min()).item()
        col_max = df.select(cast_expr.max()).item()

        online_n = 0
        online_mean = 0.0
        online_m2 = 0.0
        online_m3 = 0.0
        online_m4 = 0.0
        sum_val = 0.0
        n_zeros = 0
        n_neg = 0

        for chunk in _chunk_iter(df, options.chunk_size):
            clean = chunk.select(cast_expr.drop_nulls().alias(col))[col]
            if len(clean) == 0:
                continue
            vals = clean.to_list()
            for x in vals:
                online_n += 1
                delta = x - online_mean
                delta_n = delta / online_n
                delta_n2 = delta_n * delta_n
                term1 = delta * delta_n * (online_n - 1)
                online_mean += delta_n
                online_m4 += term1 * delta_n2 * (online_n * online_n - 3 * online_n + 3) + 6 * delta_n2 * online_m2 - 4 * delta_n * online_m3
                online_m3 += term1 * delta_n * (online_n - 2) - 3 * delta_n * online_m2
                online_m2 += term1
                sum_val += x
            n_zeros += int((clean == 0).sum())
            n_neg += int((clean < 0).sum())

        variance = online_m2 / online_n if online_n > 1 else 0.0
        std = math.sqrt(variance) if variance > 0 else 0.0
        skewness = None
        kurtosis = None
        if online_n >= 3 and std > 0:
            skewness = round((online_m3 / online_n) / (std ** 3), 6)
        if online_n >= 4 and std > 0:
            kurtosis = round((online_m4 / online_n) / (variance ** 2) - 3, 6)

        result["min"] = col_min
        result["max"] = col_max
        result["mean"] = round(online_mean, 6)
        result["std"] = round(std, 6)
        result["variance"] = round(variance, 6)
        result["skewness"] = skewness
        result["kurtosis"] = kurtosis
        result["n_zeros"] = n_zeros
        result["n_negatives"] = n_neg

        if options.mode == "full":
            pcts: dict[str, float | None] = {}
            non_null_series = df.select(cast_expr.drop_nulls().alias(col))[col]
            for q in options.numeric_percentiles:
                try:
                    val = non_null_series.quantile(q, interpolation="linear")
                    pcts[str(q)] = round(val, 6) if val is not None else None
                except Exception:
                    pcts[str(q)] = None
            result["median"] = pcts.get("0.5")
            result["percentiles"] = pcts

            if col_min is not None and col_max is not None and col_min != col_max:
                bin_width = (col_max - col_min) / options.histogram_bin_count
                edges = [col_min + i * bin_width for i in range(options.histogram_bin_count + 1)]
                bin_counts = [0] * options.histogram_bin_count

                for chunk in _chunk_iter(df, options.chunk_size):
                    clean = chunk.select(cast_expr.drop_nulls().alias(col))[col].to_list()
                    for x in clean:
                        idx = int((x - col_min) / bin_width)
                        idx = min(idx, options.histogram_bin_count - 1)
                        bin_counts[idx] += 1

                result["histogram"] = [
                    {
                        "bin_start": round(edges[i], 6),
                        "bin_end": round(edges[i + 1], 6),
                        "count": bin_counts[i],
                    }
                    for i in range(options.histogram_bin_count)
                ]

    elif _is_datetime(dtype) and (n_rows - null_count) > 0:
        result["min"] = str(df[col].min())
        result["max"] = str(df[col].max())

    if exact_distinct <= options.cardinality_limit or dtype_label in ("boolean", "categorical"):
        try:
            vc = df[col].value_counts(sort=True).head(options.cardinality_limit)
            result["top_values"] = [
                {
                    "value": row[col],
                    "count": row["count"],
                    "pct": round(row["count"] / n_rows, 6) if n_rows > 0 else 0.0,
                }
                for row in vc.iter_rows(named=True)
            ]
        except Exception:
            result["top_values"] = []

    if options.mode == "full" and options.detect_patterns and dtype_label == "string":
        sample_vals = df[col].drop_nulls().head(options.pattern_sample_size).to_list()
        sample_n = len(sample_vals)
        if sample_n > 0:
            pattern_results: list[dict[str, Any]] = []
            for pat_name, pat_re in _PATTERNS.items():
                count = sum(1 for v in sample_vals if v and pat_re.match(str(v)))
                if count > 0:
                    pattern_results.append({
                        "pattern": pat_name,
                        "count": count,
                        "pct": round(count / sample_n, 4),
                        "sampled_from": sample_n,
                    })
            result["pattern_matches"] = pattern_results

    warnings: list[dict[str, str]] = []

    if null_pct > 0.3:
        warnings.append({"code": WARNING_HIGH_NULL, "column": col, "severity": "warn"})
    if exact_distinct == 1 and n_rows > 1:
        warnings.append({"code": WARNING_CONSTANT, "column": col, "severity": "error"})
    if is_unique and n_rows > 100:
        warnings.append({"code": WARNING_IDENTIFIER, "column": col, "severity": "info"})
    if dtype_label in ("string", "categorical") and exact_distinct > options.cardinality_limit and null_pct < 0.5:
        warnings.append({"code": WARNING_HIGH_CARDINALITY, "column": col, "severity": "warn"})
    if result.get("skewness") is not None and abs(result["skewness"]) > 3:
        warnings.append({"code": WARNING_SKEWED, "column": col, "severity": "info"})
    if _is_numeric(dtype) and result.get("n_zeros") == (n_rows - null_count) and (n_rows - null_count) > 0:
        warnings.append({"code": WARNING_ALL_ZERO, "column": col, "severity": "warn"})

    result["warnings"] = warnings
    return result


def compute_summary(df: pl.DataFrame, options: ProfileOptions) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    total_nulls = 0
    dup_count = 0

    for chunk in _chunk_iter(df, options.chunk_size):
        for col in chunk.columns:
            total_nulls += chunk[col].null_count()

    for chunk in _chunk_iter(df, options.chunk_size):
        dup_count += int(chunk.is_duplicated().sum())

    n_numeric = sum(1 for c in df.columns if _is_numeric(df[c].dtype))
    n_datetime = sum(1 for c in df.columns if _is_datetime(df[c].dtype))
    n_categorical = n_cols - n_numeric - n_datetime

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "total_null_pct": round(total_nulls / total_cells, 6) if total_cells > 0 else 0.0,
        "duplicate_rows": dup_count,
        "duplicate_pct": round(dup_count / n_rows, 6) if n_rows > 0 else 0.0,
        "memory_mb": round(df.estimated_size("mb"), 4),
        "column_types": {
            "numeric": n_numeric,
            "categorical": n_categorical,
            "datetime": n_datetime,
        },
    }


def compute_correlations(
    df: pl.DataFrame,
    options: ProfileOptions,
) -> list[dict[str, Any]]:
    if options.mode != "full" or options.correlation_method == "none":
        return []

    numeric_cols = [c for c in df.columns if _is_numeric(df[c].dtype)]
    if len(numeric_cols) < 2:
        return []

    n = len(df)

    if options.correlation_method == "pearson":
        sums: dict[str, float] = {c: 0.0 for c in numeric_cols}
        sum_sq: dict[str, float] = {c: 0.0 for c in numeric_cols}
        sum_cross: dict[tuple[str, str], float] = {}
        count: dict[tuple[str, str], int] = {}

        for i, ca in enumerate(numeric_cols):
            for cb in numeric_cols[i + 1:]:
                sum_cross[(ca, cb)] = 0.0
                count[(ca, cb)] = 0

        chunk_n = 0
        for chunk in _chunk_iter(df, options.chunk_size):
            chunk_n += len(chunk)
            for c in numeric_cols:
                vals = chunk[c].drop_nulls().cast(pl.Float64).to_list()
                for v in vals:
                    sums[c] += v
                    sum_sq[c] += v * v

            for i, ca in enumerate(numeric_cols):
                for cb in numeric_cols[i + 1:]:
                    pair = chunk.select([ca, cb]).drop_nulls()
                    if len(pair) == 0:
                        continue
                    a_vals = pair[ca].cast(pl.Float64).to_list()
                    b_vals = pair[cb].cast(pl.Float64).to_list()
                    for a, b in zip(a_vals, b_vals):
                        sum_cross[(ca, cb)] += a * b
                        count[(ca, cb)] += 1

        pairs: list[dict[str, Any]] = []
        for i, ca in enumerate(numeric_cols):
            for cb in numeric_cols[i + 1:]:
                k = (ca, cb)
                cnt = count[k]
                if cnt < 2:
                    continue
                mean_a = sums[ca] / n
                mean_b = sums[cb] / n
                cov = sum_cross[k] / cnt - mean_a * mean_b
                var_a = sum_sq[ca] / n - mean_a ** 2
                var_b = sum_sq[cb] / n - mean_b ** 2
                if var_a <= 0 or var_b <= 0:
                    continue
                corr = cov / math.sqrt(var_a * var_b)
                corr = max(-1.0, min(1.0, corr))
                if abs(corr) >= options.correlation_threshold:
                    pairs.append({
                        "col_a": ca,
                        "col_b": cb,
                        "method": "pearson",
                        "value": round(corr, 4),
                    })
        return pairs

    if options.correlation_method == "spearman":
        sample_df = df.select(numeric_cols)
        if len(sample_df) > options.sample_size:
            sample_df = sample_df.sample(n=options.sample_size, seed=42)

        ranked = sample_df.select([
            pl.col(c).rank(method="average").alias(c)
            for c in numeric_cols
        ])

        pairs = []
        for i, ca in enumerate(numeric_cols):
            for cb in numeric_cols[i + 1:]:
                try:
                    val = ranked.select(pl.corr(ca, cb)).item()
                    if val is not None and abs(val) >= options.correlation_threshold:
                        pairs.append({
                            "col_a": ca,
                            "col_b": cb,
                            "method": "spearman",
                            "value": round(val, 4),
                        })
                except Exception:
                    pass
        return pairs

    return []


def collect_dataset_warnings(
    summary: dict[str, Any],
    column_profiles: list[dict[str, Any]],
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []

    if summary["duplicate_pct"] > 0.05:
        warnings.append({
            "code": WARNING_HIGH_DUPLICATE,
            "column": "__dataset__",
            "severity": "warn",
        })

    for cp in column_profiles:
        warnings.extend(cp.get("warnings", []))

    return warnings