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
class ClassifyResult:
    columns: list[ColumnClassification] = field(default_factory=list)
    dataset_domain: Optional[str] = None
    dataset_purpose: Optional[str] = None

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
        }


@dataclass
class ClassifyConfig:
    mode: str = "hybrid"
    confidence_threshold: float = 0.0
    overrides: dict[str, str] = field(default_factory=dict)