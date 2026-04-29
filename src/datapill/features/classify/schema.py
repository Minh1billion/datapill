from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SemanticType(str, Enum):
    IDENTIFIER = "identifier"
    NUMERICAL_CONTINUOUS = "numerical_continuous"
    NUMERICAL_DISCRETE = "numerical_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    TEXT_FREEFORM = "text_freeform"
    TEXT_STRUCTURED = "text_structured"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    GEOSPATIAL = "geospatial"
    EMBEDDING = "embedding"
    TARGET_LABEL = "target_label"
    UNKNOWN = "unknown"


@dataclass
class ColumnClassification:
    name: str
    semantic_type: SemanticType
    confidence: float
    reasoning: str
    source: str
    overridden: bool = False


@dataclass
class ProfileSignals:
    null_pct: float = 0.0
    distinct_count: int = 0
    distinct_pct: float = 0.0
    is_unique: bool = False
    skewness: Optional[float] = None
    pattern_names: list[str] = field(default_factory=list)
    top_value_count: Optional[int] = None

    @classmethod
    def from_profile_column(cls, col: dict) -> "ProfileSignals":
        pattern_matches = col.get("pattern_matches") or []
        pattern_names = [p["pattern"] for p in pattern_matches if p.get("pct", 0) >= 0.5]

        distinct_count = col.get("distinct_count", 0)

        return cls(
            null_pct=col.get("null_pct", 0.0),
            distinct_count=distinct_count,
            distinct_pct=col.get("distinct_pct", 0.0),
            is_unique=col.get("is_unique", False),
            skewness=col.get("skewness"),
            pattern_names=pattern_names,
            top_value_count=distinct_count if distinct_count > 0 else None,
        )


def extract_profile_signals(profile_data: dict) -> dict[str, "ProfileSignals"]:
    signals: dict[str, ProfileSignals] = {}
    for col in profile_data.get("columns", []):
        name = col.get("name")
        if name:
            signals[name] = ProfileSignals.from_profile_column(col)
    return signals


@dataclass
class ClassifyResult:
    columns: list[ColumnClassification] = field(default_factory=list)
    dataset_domain: Optional[str] = None
    dataset_purpose: Optional[str] = None
    profile_used: bool = False

    def to_dict(self) -> dict:
        return {
            "columns": [
                {
                    "name": c.name,
                    "semantic_type": c.semantic_type.value,
                    "confidence": c.confidence,
                    "reasoning": c.reasoning,
                    "source": c.source,
                    "overridden": c.overridden,
                }
                for c in self.columns
            ],
            "dataset_domain": self.dataset_domain,
            "dataset_purpose": self.dataset_purpose,
            "profile_used": self.profile_used,
        }


@dataclass
class ClassifyConfig:
    mode: str = "hybrid"
    confidence_threshold: float = 0.0
    overrides: dict[str, str] = field(default_factory=dict)