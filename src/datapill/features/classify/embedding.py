import numpy as np
import polars as pl
from typing import Optional

from .schema import SemanticType, ColumnClassification, ProfileSignals
from .rule_based import _apply_profile_signals

_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_MARGIN_THRESHOLD = 0.04

_LOW_CARDINALITY_THRESHOLD = 20
_IDENTIFIER_UNIQUE_THRESHOLD = 0.95


def _classify_from_profile_only(
    col_name: str,
    signals: ProfileSignals,
) -> Optional[ColumnClassification]:
    reasoning_parts: list[str] = []

    if signals.pattern_names:
        from .rule_based import _PATTERN_TO_SEMANTIC
        for pattern_name in signals.pattern_names:
            mapped = _PATTERN_TO_SEMANTIC.get(pattern_name)
            if mapped is not None:
                return ColumnClassification(
                    name=col_name,
                    semantic_type=mapped,
                    confidence=0.80,
                    reasoning=f"profile-only: {pattern_name} pattern in ≥50% of values",
                    source="embedding",
                )

    if signals.is_unique and signals.distinct_count > 100:
        return ColumnClassification(
            name=col_name,
            semantic_type=SemanticType.IDENTIFIER,
            confidence=0.75,
            reasoning="profile-only: all values unique with high cardinality",
            source="embedding",
        )

    if (
        signals.top_value_count is not None
        and signals.top_value_count <= _LOW_CARDINALITY_THRESHOLD
        and not signals.is_unique
    ):
        return ColumnClassification(
            name=col_name,
            semantic_type=SemanticType.CATEGORICAL_NOMINAL,
            confidence=0.70,
            reasoning=f"profile-only: low cardinality ({signals.top_value_count} distinct values)",
            source="embedding",
        )

    if signals.distinct_count > _LOW_CARDINALITY_THRESHOLD and signals.distinct_pct < _IDENTIFIER_UNIQUE_THRESHOLD:
        if signals.skewness is not None and abs(signals.skewness) > 3:
            return ColumnClassification(
                name=col_name,
                semantic_type=SemanticType.NUMERICAL_CONTINUOUS,
                confidence=0.65,
                reasoning=f"profile-only: high cardinality + high skewness ({signals.skewness:.2f})",
                source="embedding",
            )
        return ColumnClassification(
            name=col_name,
            semantic_type=SemanticType.NUMERICAL_DISCRETE,
            confidence=0.60,
            reasoning=f"profile-only: high cardinality ({signals.distinct_count} distinct values)",
            source="embedding",
        )

    return None

_ANCHOR_TEXTS: dict[SemanticType, list[str]] = {
    SemanticType.IDENTIFIER: [
        "unique row identifier surrogate key primary key auto increment id",
        "record id user id order id foreign key reference sequential integer id",
        "uuid guid hash fingerprint row number index column",
    ],
    SemanticType.NUMERICAL_CONTINUOUS: [
        "price cost amount revenue salary wage income budget float decimal money",
        "weight height temperature distance ratio percentage score measurement",
        "continuous real-valued metric sensor reading physical quantity",
    ],
    SemanticType.NUMERICAL_DISCRETE: [
        "count total quantity number of items purchases visits occurrences integer",
        "stock inventory units pieces frequency how many whole number tally",
        "discrete countable non-negative integer events transactions clicks",
    ],
    SemanticType.CATEGORICAL_NOMINAL: [
        "category type status group class label tag unordered string nominal",
        "department country brand color product type segment gender mode",
        "named group with no inherent rank or ordering between values",
    ],
    SemanticType.CATEGORICAL_ORDINAL: [
        "priority rank order grade severity tier level ordinal ranked position",
        "low medium high small large first second third bronze silver gold",
        "ordered category with meaningful progression between values",
    ],
    SemanticType.TEXT_FREEFORM: [
        "description comment note review feedback narrative long text paragraph",
        "message body content remark summary bio abstract free-form unstructured",
        "user generated text open ended response natural language string",
    ],
    SemanticType.TEXT_STRUCTURED: [
        "email address phone number url link postal code zip ssn formatted string",
        "structured text following a fixed pattern or format with validation rules",
        "domain name ip address credit card barcode iban regex constrained field",
    ],
    SemanticType.DATETIME: [
        "date time timestamp created updated born expires scheduled datetime",
        "year month day hour minute second week quarter calendar time point",
        "event time log time audit timestamp ISO 8601 unix epoch temporal value",
    ],
    SemanticType.BOOLEAN: [
        "true false yes no binary flag indicator switch on off toggle bit",
        "is active has verified was deleted can approve boolean condition",
        "binary state two possible values mutually exclusive true or false",
    ],
    SemanticType.GEOSPATIAL: [
        "latitude longitude coordinates location point on map geographic place",
        "city country region state province address spatial area geo boundary",
        "GPS coordinate place name geometry polygon bounding box map layer",
    ],
    SemanticType.EMBEDDING: [
        "dense vector embedding representation neural network output feature space",
        "high dimensional float array latent space encoding similarity search",
        "word2vec sentence transformer BERT output vector store embedding column",
    ],
    SemanticType.TARGET_LABEL: [
        "target label outcome class prediction dependent variable ground truth y",
        "response variable supervised learning label what we are trying to predict",
        "binary label multiclass output regression target churn conversion fraud",
    ],
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _build_column_text(
    col_name: str,
    dtype: pl.DataType,
    series: pl.Series,
    profile_signals: Optional[ProfileSignals] = None,
) -> str:
    non_null = series.drop_nulls()
    top_values: list[str] = []
    sample_value = ""

    if len(non_null) > 0:
        try:
            value_counts = non_null.value_counts(sort=True)
            top_n = min(3, len(value_counts))
            top_values = [str(value_counts[i, 0]) for i in range(top_n)]
        except Exception:
            top_values = [str(v) for v in non_null.head(3).to_list()]
        sample_value = str(non_null[0])

    parts = [
        col_name.replace("_", " ").replace("-", " "),
        str(dtype),
        " ".join(top_values),
        sample_value,
    ]

    if profile_signals is not None:
        hints: list[str] = []
        if profile_signals.pattern_names:
            hints.append(" ".join(profile_signals.pattern_names))
        if profile_signals.is_unique:
            hints.append("unique identifier")
        if profile_signals.top_value_count is not None and profile_signals.top_value_count <= 20:
            hints.append("low cardinality categorical")
        if profile_signals.skewness is not None and abs(profile_signals.skewness) > 3:
            hints.append("skewed continuous numeric")
        if hints:
            parts.append(" ".join(hints))

    return " ".join(p for p in parts if p).strip()


def _encode(model, texts: list[str]) -> np.ndarray:
    return np.array(list(model.embed(texts)), dtype=np.float32)


class EmbeddingClassifier:
    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._model = None
        self._anchor_embeddings: Optional[dict[SemanticType, np.ndarray]] = None
        self._cache_dir = cache_dir

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "Embedding mode requires fastembed. "
                "Install with: pip install datapill[ml]"
            )
        self._model = TextEmbedding(
            model_name=_MODEL_NAME,
            cache_dir=self._cache_dir,
            show_progress=False,
        )

    def _build_anchors(self):
        if self._anchor_embeddings is not None:
            return
        self._load_model()
        self._anchor_embeddings = {}
        for semantic_type, texts in _ANCHOR_TEXTS.items():
            embeddings = _encode(self._model, texts)
            self._anchor_embeddings[semantic_type] = embeddings.mean(axis=0)

    def _score(self, col_embedding: np.ndarray) -> tuple[SemanticType, float, float]:
        best_type = SemanticType.UNKNOWN
        best_sim = -1.0
        second_sim = -1.0

        for semantic_type, anchor_vec in self._anchor_embeddings.items():
            sim = _cosine_similarity(col_embedding, anchor_vec)
            if sim > best_sim:
                second_sim = best_sim
                best_sim = sim
                best_type = semantic_type
            elif sim > second_sim:
                second_sim = sim

        margin = best_sim - second_sim
        return best_type, best_sim, margin

    def _make_result(
        self,
        col_name: str,
        best_type: SemanticType,
        best_sim: float,
        margin: float,
        profile_signals: Optional[ProfileSignals],
    ) -> ColumnClassification:
        if margin < _MARGIN_THRESHOLD:
            if profile_signals is not None:
                profile_result = _classify_from_profile_only(col_name, profile_signals)
                if profile_result is not None:
                    return profile_result
            return ColumnClassification(
                name=col_name,
                semantic_type=SemanticType.UNKNOWN,
                confidence=0.0,
                reasoning=f"embedding margin too low ({margin:.3f}) - ambiguous",
                source="embedding",
            )

        raw_confidence = 0.70 + (best_sim - 0.5) * 0.4 + margin * 0.2
        confidence = float(np.clip(raw_confidence, 0.50, 0.90))

        result = ColumnClassification(
            name=col_name,
            semantic_type=best_type,
            confidence=confidence,
            reasoning=f"embedding similarity {best_sim:.3f} to {best_type.value} anchor (margin {margin:.3f})",
            source="embedding",
        )

        if profile_signals is not None:
            return _apply_profile_signals(result, profile_signals)

        return result

    def classify_column(
        self,
        col_name: str,
        dtype: pl.DataType,
        series: pl.Series,
        profile_signals: Optional[ProfileSignals] = None,
    ) -> ColumnClassification:
        self._build_anchors()
        col_text = _build_column_text(col_name, dtype, series, profile_signals)
        col_embedding = _encode(self._model, [col_text])[0]
        best_type, best_sim, margin = self._score(col_embedding)
        return self._make_result(col_name, best_type, best_sim, margin, profile_signals)

    def classify_batch(
        self,
        columns: list[tuple[str, pl.DataType, pl.Series]],
        profile_signals_map: Optional[dict[str, ProfileSignals]] = None,
    ) -> list[ColumnClassification]:
        self._build_anchors()

        texts = [
            _build_column_text(
                name, dtype, series,
                profile_signals_map.get(name) if profile_signals_map else None,
            )
            for name, dtype, series in columns
        ]
        col_embeddings = _encode(self._model, texts)

        results = []
        for i, (col_name, dtype, series) in enumerate(columns):
            best_type, best_sim, margin = self._score(col_embeddings[i])
            signals = profile_signals_map.get(col_name) if profile_signals_map else None
            results.append(self._make_result(col_name, best_type, best_sim, margin, signals))

        return results


_shared_classifier: Optional[EmbeddingClassifier] = None


def get_embedding_classifier(cache_dir: Optional[str] = None) -> EmbeddingClassifier:
    global _shared_classifier
    if _shared_classifier is None:
        _shared_classifier = EmbeddingClassifier(cache_dir=cache_dir)
    elif cache_dir and _shared_classifier._cache_dir != cache_dir:
        _shared_classifier = EmbeddingClassifier(cache_dir=cache_dir)
    return _shared_classifier