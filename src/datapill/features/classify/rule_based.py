import re
from typing import Optional
import polars as pl

from .schema import SemanticType, ColumnClassification, ProfileSignals

_CERTAIN_CONFIDENCE = 1.0
_HEURISTIC_CONFIDENCE = 0.80

_DTYPE_CERTAIN: list[tuple[tuple, SemanticType, str]] = [
    ((pl.Boolean,), SemanticType.BOOLEAN, "dtype Boolean"),
    ((pl.Date, pl.Datetime, pl.Time, pl.Duration), SemanticType.DATETIME, "dtype is temporal"),
    ((pl.Float32, pl.Float64), SemanticType.NUMERICAL_CONTINUOUS, "dtype Float"),
]

_NAME_CERTAIN: list[tuple[re.Pattern, SemanticType, str]] = [
    (re.compile(r"\b(uuid|guid)\b", re.I),                                        SemanticType.IDENTIFIER,           "name: uuid/guid"),
    (re.compile(r"(^|_)id$", re.I),                                               SemanticType.IDENTIFIER,           "name: id/_id suffix"),
    (re.compile(r"^id_", re.I),                                                    SemanticType.IDENTIFIER,           "name: id_ prefix"),
    (re.compile(r"\b(email|e_mail)\b", re.I),                                     SemanticType.TEXT_STRUCTURED,      "name: email"),
    (re.compile(r"\b(phone|tel|mobile|cellphone)\b", re.I),                       SemanticType.TEXT_STRUCTURED,      "name: phone"),
    (re.compile(r"\b(url|href|uri)\b", re.I),                                     SemanticType.TEXT_STRUCTURED,      "name: url/href/uri"),
    (re.compile(r"\b(zip|zipcode|postal|postcode|iban|ssn|npi)\b", re.I),         SemanticType.TEXT_STRUCTURED,      "name: structured code field"),
    (re.compile(r"\b(lat|latitude)\b", re.I),                                     SemanticType.GEOSPATIAL,           "name: latitude"),
    (re.compile(r"\b(lon|lng|longitude)\b", re.I),                                SemanticType.GEOSPATIAL,           "name: longitude"),
    (re.compile(r"\b(embedding|embed|vector|vec)\b", re.I),                       SemanticType.EMBEDDING,            "name: embedding/vector"),
    (re.compile(r"(^|_)(at|date|datetime|timestamp)$", re.I),                     SemanticType.DATETIME,             "name: _at/_date/_datetime/_timestamp suffix"),
    (re.compile(r"^(is|has|can|was|did|should|will)_", re.I),                     SemanticType.BOOLEAN,              "name: boolean prefix"),
    (re.compile(r"_(flag|enabled|active|deleted|verified|approved)$", re.I),      SemanticType.BOOLEAN,              "name: boolean suffix"),
]

_NAME_HEURISTIC: list[tuple[re.Pattern, SemanticType, str]] = [
    (re.compile(r"\b(price|cost|amount|revenue|salary|wage|income|fee|rate|budget|balance|subtotal|discount|tax)\b", re.I),
                                                                                   SemanticType.NUMERICAL_CONTINUOUS, "name: monetary/rate field"),
    (re.compile(r"\b(weight|height|width|depth|length|volume|distance|speed|temperature|ratio|score|pct|percent|percentage)\b", re.I),
                                                                                   SemanticType.NUMERICAL_CONTINUOUS, "name: physical/ratio measurement"),
    (re.compile(r"\b(age|duration|tenure)\b", re.I),                              SemanticType.NUMERICAL_CONTINUOUS, "name: duration/age field"),
    (re.compile(r"\b(qty|quantity)\b", re.I),                                     SemanticType.NUMERICAL_DISCRETE,   "name: quantity field"),
    (re.compile(r"(^|_)(count|num|total|n)(_|$)", re.I),                          SemanticType.NUMERICAL_DISCRETE,   "name: count/num prefix or suffix"),
    (re.compile(r"\b(rank|ranking|position|seq|sequence)\b", re.I),               SemanticType.CATEGORICAL_ORDINAL,  "name: rank/order field"),
    (re.compile(r"\b(priority|severity|tier|grade)\b", re.I),                     SemanticType.CATEGORICAL_ORDINAL,  "name: ordinal level field"),
    (re.compile(r"\b(category|cat|type|kind|class|group|segment|tag|genre|brand|status|state|mode|role|department|division)\b", re.I),
                                                                                   SemanticType.CATEGORICAL_NOMINAL,  "name: category/type field"),
    (re.compile(r"\b(gender|sex|country|region|city|nationality|language|currency|platform|channel|source|medium|device)\b", re.I),
                                                                                   SemanticType.CATEGORICAL_NOMINAL,  "name: demographic/dimension field"),
    (re.compile(r"\b(description|desc|comment|note|remark|bio|summary|message|body|content|narrative|feedback|review)\b", re.I),
                                                                                   SemanticType.TEXT_FREEFORM,        "name: free-text field"),
    (re.compile(r"(^|_)(year|month|day|hour|minute|second|week|quarter)(_|$)", re.I),
                                                                                   SemanticType.DATETIME,             "name: calendar component"),
]

_PATTERN_TO_SEMANTIC: dict[str, SemanticType] = {
    "email":    SemanticType.TEXT_STRUCTURED,
    "url":      SemanticType.TEXT_STRUCTURED,
    "phone_vn": SemanticType.TEXT_STRUCTURED,
    "uuid":     SemanticType.IDENTIFIER,
    "date_iso": SemanticType.DATETIME,
}

_LOW_CARDINALITY_THRESHOLD = 20
_IDENTIFIER_UNIQUE_THRESHOLD = 0.95


def _classify_certain(col_name: str, dtype: pl.DataType) -> Optional[ColumnClassification]:
    for dtypes, semantic_type, reasoning in _DTYPE_CERTAIN:
        if dtype in dtypes:
            return ColumnClassification(
                name=col_name,
                semantic_type=semantic_type,
                confidence=_CERTAIN_CONFIDENCE,
                reasoning=f"certain: {reasoning}",
                source="rule_based",
            )

    for pattern, semantic_type, reasoning in _NAME_CERTAIN:
        if pattern.search(col_name):
            return ColumnClassification(
                name=col_name,
                semantic_type=semantic_type,
                confidence=_CERTAIN_CONFIDENCE,
                reasoning=f"certain: {reasoning}",
                source="rule_based",
            )

    for pattern, semantic_type, reasoning in _NAME_HEURISTIC:
        if pattern.search(col_name):
            return ColumnClassification(
                name=col_name,
                semantic_type=semantic_type,
                confidence=_HEURISTIC_CONFIDENCE,
                reasoning=f"heuristic: {reasoning}",
                source="rule_based",
            )

    return None


def _apply_profile_signals(
    result: ColumnClassification,
    signals: ProfileSignals,
) -> ColumnClassification:
    if result.confidence >= _CERTAIN_CONFIDENCE:
        notes: list[str] = []
        if signals.null_pct > 0.3:
            notes.append(f"profile: high null rate ({signals.null_pct:.0%})")
        if not notes:
            return result
        return ColumnClassification(
            name=result.name,
            semantic_type=result.semantic_type,
            confidence=result.confidence,
            reasoning="; ".join([result.reasoning] + notes),
            source=result.source,
            overridden=result.overridden,
        )

    semantic_type = result.semantic_type
    confidence = result.confidence
    reasoning_parts = [result.reasoning]

    if signals.pattern_names:
        for pattern_name in signals.pattern_names:
            mapped = _PATTERN_TO_SEMANTIC.get(pattern_name)
            if mapped is not None:
                if semantic_type != mapped:
                    semantic_type = mapped
                    confidence = max(confidence, 0.75)
                    reasoning_parts.append(f"profile: {pattern_name} pattern in ≥50% of values")
                else:
                    confidence = min(confidence + 0.05, 0.95)
                    reasoning_parts.append(f"profile: {pattern_name} pattern confirmed")
                break

    if signals.is_unique and signals.distinct_count > 100:
        if semantic_type not in (SemanticType.IDENTIFIER, SemanticType.TEXT_STRUCTURED, SemanticType.DATETIME):
            semantic_type = SemanticType.IDENTIFIER
            confidence = max(confidence, 0.70)
            reasoning_parts.append("profile: all values unique with high cardinality")
        elif semantic_type == SemanticType.IDENTIFIER:
            confidence = min(confidence + 0.08, 0.95)
            reasoning_parts.append("profile: uniqueness confirms identifier")

    elif (
        signals.distinct_pct >= _IDENTIFIER_UNIQUE_THRESHOLD
        and not signals.is_unique
        and signals.distinct_count > 100
        and semantic_type == SemanticType.IDENTIFIER
    ):
        confidence = min(confidence + 0.05, 0.95)
        reasoning_parts.append(f"profile: near-unique ({signals.distinct_pct:.0%} distinct)")

    if (
        signals.top_value_count is not None
        and signals.top_value_count <= _LOW_CARDINALITY_THRESHOLD
        and semantic_type not in (SemanticType.BOOLEAN, SemanticType.IDENTIFIER, SemanticType.DATETIME)
        and not signals.is_unique
    ):
        if semantic_type not in (SemanticType.CATEGORICAL_NOMINAL, SemanticType.CATEGORICAL_ORDINAL):
            semantic_type = SemanticType.CATEGORICAL_NOMINAL
            confidence = max(confidence, 0.65)
            reasoning_parts.append(f"profile: low cardinality ({signals.top_value_count} distinct values)")
        else:
            confidence = min(confidence + 0.05, 0.95)
            reasoning_parts.append(f"profile: low cardinality ({signals.top_value_count} distinct values) confirms categorical")

    if (
        signals.skewness is not None
        and abs(signals.skewness) > 3
        and semantic_type in (SemanticType.NUMERICAL_DISCRETE, SemanticType.NUMERICAL_CONTINUOUS)
    ):
        if semantic_type != SemanticType.NUMERICAL_CONTINUOUS:
            semantic_type = SemanticType.NUMERICAL_CONTINUOUS
            confidence = max(confidence, 0.65)
            reasoning_parts.append(f"profile: high skewness ({signals.skewness:.2f}) suggests continuous")
        else:
            reasoning_parts.append(f"profile: high skewness ({signals.skewness:.2f})")

    if signals.null_pct > 0.3:
        reasoning_parts.append(f"profile: high null rate ({signals.null_pct:.0%})")

    return ColumnClassification(
        name=result.name,
        semantic_type=semantic_type,
        confidence=confidence,
        reasoning="; ".join(reasoning_parts),
        source=result.source,
        overridden=result.overridden,
    )


def classify_column_rule_based(
    col_name: str,
    dtype: pl.DataType,
    series: pl.Series,
    profile_signals: Optional[ProfileSignals] = None,
) -> Optional[ColumnClassification]:
    result = _classify_certain(col_name, dtype)

    if result is None:
        return None

    if profile_signals is not None:
        return _apply_profile_signals(result, profile_signals)

    return result