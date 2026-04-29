import re
from typing import Any
import polars as pl

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_DATETIME_DTYPES = (pl.Date, pl.Datetime, pl.Duration, pl.Time)

_DTYPE_MAP: dict[str, str] = {
    "Int8": "integer", "Int16": "integer", "Int32": "integer", "Int64": "integer",
    "UInt8": "integer", "UInt16": "integer", "UInt32": "integer", "UInt64": "integer",
    "Float32": "float", "Float64": "float",
    "Boolean": "boolean",
    "Utf8": "string", "String": "string", "Categorical": "categorical",
    "Date": "datetime", "Datetime": "datetime", "Time": "datetime", "Duration": "datetime",
    "List": "list", "Array": "array", "Struct": "struct",
    "Binary": "binary",
}


def _infer_dtype(dtype: pl.DataType) -> str:
    if isinstance(dtype, pl.Decimal):
        return "float"
    if isinstance(dtype, (pl.Datetime, pl.Date, pl.Duration, pl.Time)):
        return "datetime"
    return _DTYPE_MAP.get(str(dtype), "unknown")

_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$"),
    "url": re.compile(r"^https?://\S+$"),
    "phone_vn": re.compile(r"^(0|\+84)[3-9]\d{8}$"),
    "date_iso": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "uuid": re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
}

_PERCENTILE_QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


class ColumnStatsComputer:
    def __init__(
        self,
        cardinality_limit: int = 50,
        text_sample_limit: int = 10,
        detect_patterns: bool = True,
        numeric_percentiles: list[float] | None = None,
        histogram_bin_count: int = 20,
    ) -> None:
        self.cardinality_limit = cardinality_limit
        self.text_sample_limit = text_sample_limit
        self.detect_patterns = detect_patterns
        self.numeric_percentiles = numeric_percentiles or _PERCENTILE_QUANTILES
        self.histogram_bin_count = histogram_bin_count

    def compute(self, series: pl.Series) -> dict[str, Any]:
        n = len(series)
        null_count = series.null_count()
        dtype = series.dtype
        dtype_str = str(dtype)
        dtype_inferred = _infer_dtype(dtype)

        clean = series.drop_nulls()
        distinct_count = series.n_unique()

        result: dict[str, Any] = {
            "name": series.name,
            "dtype_physical": dtype_str,
            "dtype_inferred": dtype_inferred,
            "null_count": null_count,
            "null_pct": round(null_count / n, 6) if n > 0 else 0.0,
            "distinct_count": distinct_count,
            "distinct_pct": round(distinct_count / n, 6) if n > 0 else 0.0,
            "is_unique": distinct_count == n and n > 0,
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
            "sample_values": clean.head(self.text_sample_limit).to_list(),
            "warnings": [],
        }

        is_numeric = dtype in _NUMERIC_DTYPES or isinstance(dtype, pl.Decimal)
        is_datetime = isinstance(dtype, (pl.Datetime, pl.Date, pl.Duration, pl.Time))

        if is_numeric and len(clean) > 0:
            numeric_clean = clean.cast(pl.Float64) if isinstance(dtype, pl.Decimal) else clean
            result.update(self._numeric_stats(numeric_clean, n))

        elif is_datetime and len(clean) > 0:
            result["min"] = str(clean.min())
            result["max"] = str(clean.max())

        if distinct_count <= self.cardinality_limit or dtype_inferred in ("string", "categorical", "boolean"):
            result["top_values"] = self._top_values(series, n)

        if self.detect_patterns and dtype_inferred == "string" and len(clean) > 0:
            result["pattern_matches"] = self._detect_patterns(clean, n)

        result["warnings"] = self._detect_warnings(series, result, n)
        return result

    def _numeric_stats(self, clean: pl.Series, total_n: int) -> dict[str, Any]:
        min_val = clean.min()
        max_val = clean.max()
        mean_val = clean.mean()

        stats: dict[str, Any] = {
            "min": min_val,
            "max": max_val,
            "mean": round(mean_val, 6) if mean_val is not None else None,
            "median": clean.median(),
            "std": clean.std(),
            "variance": clean.var(),
        }

        try:
            stats["skewness"] = clean.skew()
        except Exception:
            stats["skewness"] = None

        try:
            stats["kurtosis"] = clean.kurtosis()
        except Exception:
            stats["kurtosis"] = None

        pcts: dict[str, float] = {}
        for q in self.numeric_percentiles:
            try:
                val = clean.quantile(q, interpolation="linear")
                pcts[str(q)] = round(val, 6) if val is not None else None
            except Exception:
                pass
        stats["percentiles"] = pcts

        if min_val is not None and max_val is not None and min_val != max_val:
            try:
                bin_width = (max_val - min_val) / self.histogram_bin_count
                edges = [min_val + i * bin_width for i in range(self.histogram_bin_count + 1)]
                cut = clean.cut(edges[1:-1], labels=[str(i) for i in range(self.histogram_bin_count)])
                counts = cut.value_counts(sort=False).sort("value")
                bins = [
                    {
                        "bin_start": round(edges[i], 6),
                        "bin_end": round(edges[i + 1], 6),
                        "count": int(counts["count"][i]) if i < len(counts) else 0,
                    }
                    for i in range(self.histogram_bin_count)
                ]
                stats["histogram"] = bins
            except Exception:
                stats["histogram"] = None

        return stats

    def _top_values(self, series: pl.Series, total_n: int) -> list[dict[str, Any]]:
        try:
            vc = series.value_counts(sort=True).head(self.cardinality_limit)
            col_name = series.name
            return [
                {
                    "value": row[col_name],
                    "count": row["count"],
                    "pct": round(row["count"] / total_n, 6) if total_n > 0 else 0.0,
                }
                for row in vc.iter_rows(named=True)
            ]
        except Exception:
            return []

    def _detect_patterns(self, clean: pl.Series, total_n: int) -> list[dict[str, Any]]:
        sample = clean.head(2000).to_list()
        results: list[dict[str, Any]] = []
        for name, pattern in _PATTERNS.items():
            try:
                count = sum(1 for v in sample if v and pattern.match(str(v)))
                if count > 0:
                    pct = round(count / len(sample), 4)
                    results.append({"pattern": name, "count": count, "pct": pct})
            except Exception:
                pass
        return results

    def _detect_warnings(self, series: pl.Series, stats: dict, n: int) -> list[str]:
        warnings: list[str] = []
        if stats["null_pct"] > 0.3:
            warnings.append("HIGH_NULL_RATE")
        if stats["distinct_count"] == 1 and n > 1:
            warnings.append("CONSTANT_COLUMN")
        if stats["is_unique"] and n > 100:
            warnings.append("POTENTIAL_IDENTIFIER")
        if (
            stats["dtype_inferred"] in ("string", "categorical")
            and stats["distinct_count"] > 100
            and stats["null_pct"] < 0.5
        ):
            warnings.append("HIGH_CARDINALITY")
        if stats["skewness"] is not None and abs(stats["skewness"]) > 3:
            warnings.append("SKEWED_DISTRIBUTION")
        return warnings


def compute_correlation_matrix(
    df: pl.DataFrame,
    method: str = "pearson",
    threshold: float = 0.5,
    max_rows: int = 500_000,
) -> list[dict[str, Any]]:
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in (pl.Int32, pl.Int64, pl.Float32, pl.Float64,
                           pl.Int8, pl.Int16, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        or isinstance(df[c].dtype, pl.Decimal)
    ]
    if len(numeric_cols) < 2:
        return []

    sample = df.select([
        pl.col(c).cast(pl.Float64).alias(c) if isinstance(df[c].dtype, pl.Decimal) else pl.col(c)
        for c in numeric_cols
    ])
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, seed=42)

    pairs: list[dict[str, Any]] = []
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1:]:
            try:
                if method == "pearson":
                    val = sample.select(pl.corr(col_a, col_b)).to_series()[0]
                elif method == "spearman":
                    ranked = sample.select([
                        pl.col(col_a).rank().alias(col_a),
                        pl.col(col_b).rank().alias(col_b),
                    ])
                    val = ranked.select(pl.corr(col_a, col_b)).to_series()[0]
                else:
                    continue
                if val is not None and abs(val) >= threshold:
                    pairs.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "method": method,
                        "value": round(val, 4),
                    })
            except Exception:
                pass
    return pairs