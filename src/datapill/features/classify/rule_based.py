import re
from typing import Optional
import polars as pl

from .schema import SemanticType, ColumnClassification

_NAME_PATTERNS: list[tuple[re.Pattern, SemanticType, float, str]] = [
    (re.compile(r"\b(uuid|guid)\b", re.I), SemanticType.IDENTIFIER, 0.70, "column name matches uuid/guid pattern"),
    (re.compile(r"(^|_)(id|key|pk|fk|ref|code)($|_)", re.I), SemanticType.IDENTIFIER, 0.65, "column name matches identifier pattern"),
    (re.compile(r"\b(email|e_mail|mail)\b", re.I), SemanticType.TEXT_STRUCTURED, 0.70, "column name matches email pattern"),
    (re.compile(r"\b(phone|tel|mobile|fax|cellphone)\b", re.I), SemanticType.TEXT_STRUCTURED, 0.70, "column name matches phone pattern"),
    (re.compile(r"\b(url|link|href|website|domain)\b", re.I), SemanticType.TEXT_STRUCTURED, 0.68, "column name matches url pattern"),
    (re.compile(r"\b(zip|zipcode|postal|postcode)\b", re.I), SemanticType.TEXT_STRUCTURED, 0.68, "column name matches postal code pattern"),
    (re.compile(r"\b(lat|latitude)\b", re.I), SemanticType.GEOSPATIAL, 0.70, "column name matches latitude pattern"),
    (re.compile(r"\b(lon|lng|longitude)\b", re.I), SemanticType.GEOSPATIAL, 0.70, "column name matches longitude pattern"),
    (re.compile(r"\b(geo|location|address|city|country|region|state|province)\b", re.I), SemanticType.GEOSPATIAL, 0.60, "column name matches geospatial pattern"),
    (re.compile(r"\b(date|time|timestamp|datetime|created_at|updated_at|deleted_at|born_at|expires_at)\b", re.I), SemanticType.DATETIME, 0.68, "column name matches datetime pattern"),
    (re.compile(r"\b(year|month|day|hour|minute|second|week|quarter)\b", re.I), SemanticType.DATETIME, 0.60, "column name matches datetime component pattern"),
    (re.compile(r"\b(price|cost|amount|revenue|salary|wage|fee|rate|budget|spend|expenditure)\b", re.I), SemanticType.NUMERICAL_CONTINUOUS, 0.68, "column name matches financial numeric pattern"),
    (re.compile(r"\b(age|score|grade|rating|rank|weight|height|temp|temperature|distance|duration)\b", re.I), SemanticType.NUMERICAL_CONTINUOUS, 0.65, "column name matches continuous numeric pattern"),
    (re.compile(r"\b(count|total|quantity|qty|num|number|n_|_n$)\b", re.I), SemanticType.NUMERICAL_DISCRETE, 0.65, "column name matches discrete count pattern"),
    (re.compile(r"\b(stock|inventory|units|items|pieces|pcs|in_stock|stock_qty)\b", re.I), SemanticType.NUMERICAL_DISCRETE, 0.65, "column name matches inventory/stock count pattern"),
    (re.compile(r"\b(name|first_name|last_name|full_name|username|nickname|firstname|lastname)\b", re.I), SemanticType.TEXT_FREEFORM, 0.65, "column name matches name pattern"),
    (re.compile(r"\b(description|desc|comment|note|notes|text|content|message|body|summary|remark)\b", re.I), SemanticType.TEXT_FREEFORM, 0.68, "column name matches free text pattern"),
    (re.compile(r"\b(category|cat|type|kind|class|group|segment|tier|level|status|state|tag)\b", re.I), SemanticType.CATEGORICAL_NOMINAL, 0.63, "column name matches category pattern"),
    (re.compile(r"\b(gender|sex|race|ethnicity|nationality|religion)\b", re.I), SemanticType.CATEGORICAL_NOMINAL, 0.67, "column name matches demographic category pattern"),
    (re.compile(r"\b(priority|severity|urgency|order|rank|grade|tier|level)\b", re.I), SemanticType.CATEGORICAL_ORDINAL, 0.63, "column name matches ordinal pattern"),
    (re.compile(r"\b(is_|has_|can_|flag|active|enabled|disabled|verified|deleted|approved)\b", re.I), SemanticType.BOOLEAN, 0.70, "column name matches boolean flag pattern"),
    (re.compile(r"\b(embedding|embed|vector|vec)\b", re.I), SemanticType.EMBEDDING, 0.70, "column name matches embedding pattern"),
    (re.compile(r"\b(target|label|y$|outcome|output|prediction|predict)\b", re.I), SemanticType.TARGET_LABEL, 0.65, "column name matches target label pattern"),
]

_LOW_CARDINALITY_THRESHOLD = 20
_IDENTIFIER_UNIQUE_THRESHOLD = 0.95


def _classify_by_dtype(col_name: str, dtype: pl.DataType, series: pl.Series) -> Optional[ColumnClassification]:
    n = len(series)
    if n == 0:
        return None

    if dtype == pl.Boolean:
        return ColumnClassification(
            name=col_name,
            semantic_type=SemanticType.BOOLEAN,
            confidence=0.70,
            reasoning="polars dtype is Boolean",
            source="rule_based",
        )

    if dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
        return ColumnClassification(
            name=col_name,
            semantic_type=SemanticType.DATETIME,
            confidence=0.70,
            reasoning=f"polars dtype is {dtype}",
            source="rule_based",
        )

    if dtype in (pl.Float32, pl.Float64):
        return ColumnClassification(
            name=col_name,
            semantic_type=SemanticType.NUMERICAL_CONTINUOUS,
            confidence=0.65,
            reasoning=f"polars dtype is {dtype}",
            source="rule_based",
        )

    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        non_null = series.drop_nulls()
        if len(non_null) == 0:
            return None
        distinct = non_null.n_unique()
        distinct_ratio = distinct / len(non_null)

        if distinct_ratio >= _IDENTIFIER_UNIQUE_THRESHOLD and distinct > 100:
            return ColumnClassification(
                name=col_name,
                semantic_type=SemanticType.IDENTIFIER,
                confidence=0.60,
                reasoning="integer dtype with near-unique values suggests identifier",
                source="rule_based",
            )

        if distinct <= _LOW_CARDINALITY_THRESHOLD:
            return ColumnClassification(
                name=col_name,
                semantic_type=SemanticType.CATEGORICAL_NOMINAL,
                confidence=0.60,
                reasoning=f"integer dtype with low cardinality ({distinct} unique values)",
                source="rule_based",
            )

        return ColumnClassification(
            name=col_name,
            semantic_type=SemanticType.NUMERICAL_DISCRETE,
            confidence=0.60,
            reasoning=f"integer dtype with moderate cardinality ({distinct} unique values)",
            source="rule_based",
        )

    if dtype == pl.Utf8 or dtype == pl.String:
        non_null = series.drop_nulls()
        if len(non_null) == 0:
            return None
        distinct = non_null.n_unique()
        if distinct <= _LOW_CARDINALITY_THRESHOLD:
            return ColumnClassification(
                name=col_name,
                semantic_type=SemanticType.CATEGORICAL_NOMINAL,
                confidence=0.55,
                reasoning=f"string dtype with low cardinality ({distinct} unique values)",
                source="rule_based",
            )

    return None


def _classify_by_name(col_name: str) -> Optional[ColumnClassification]:
    for pattern, semantic_type, confidence, reasoning in _NAME_PATTERNS:
        if pattern.search(col_name):
            return ColumnClassification(
                name=col_name,
                semantic_type=semantic_type,
                confidence=confidence,
                reasoning=reasoning,
                source="rule_based",
            )
    return None


def classify_column_rule_based(
    col_name: str,
    dtype: pl.DataType,
    series: pl.Series,
) -> ColumnClassification:
    dtype_result = _classify_by_dtype(col_name, dtype, series)
    name_result = _classify_by_name(col_name)

    if dtype_result and name_result:
        if name_result.confidence >= dtype_result.confidence:
            return name_result
        return dtype_result

    if dtype_result:
        return dtype_result

    if name_result:
        return name_result

    return ColumnClassification(
        name=col_name,
        semantic_type=SemanticType.UNKNOWN,
        confidence=0.0,
        reasoning="no rule matched",
        source="rule_based",
    )