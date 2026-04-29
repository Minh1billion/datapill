import numpy as np
import polars as pl
from typing import Optional

from .schema import SemanticType, ColumnClassification

_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # ~130MB ONNX, no torch required
_AMBIGUOUS_THRESHOLD = 0.70

_ANCHOR_TEXTS: dict[SemanticType, list[str]] = {
    SemanticType.IDENTIFIER: [
        "unique id identifier primary key uuid record id user id",
        "auto increment id surrogate key foreign key reference id",
    ],
    SemanticType.NUMERICAL_CONTINUOUS: [
        "price cost amount revenue salary weight height temperature float decimal",
        "continuous numeric value measurement ratio interval scale",
    ],
    SemanticType.NUMERICAL_DISCRETE: [
        "count quantity integer number of items total purchases frequency",
        "discrete count whole number occurrences visits transactions",
    ],
    SemanticType.CATEGORICAL_NOMINAL: [
        "category type status group class label string nominal no order",
        "department country product type color brand segment unordered",
    ],
    SemanticType.CATEGORICAL_ORDINAL: [
        "priority level rank order grade severity tier ordinal ranked",
        "low medium high small large first second third ordered category",
    ],
    SemanticType.TEXT_FREEFORM: [
        "description comment note review feedback free text long string",
        "message body content remark narrative unstructured text",
    ],
    SemanticType.TEXT_STRUCTURED: [
        "email phone url address postal code formatted pattern structured",
        "email address telephone number website link zip code ssn",
    ],
    SemanticType.DATETIME: [
        "date time timestamp created updated born expires datetime",
        "year month day hour minute second week quarter date column",
    ],
    SemanticType.BOOLEAN: [
        "is has flag active enabled verified true false binary yes no",
        "boolean flag indicator switch toggle on off approved deleted",
    ],
    SemanticType.GEOSPATIAL: [
        "latitude longitude location address city country region coordinates geo",
        "spatial geographic point area map coordinates place location",
    ],
    SemanticType.EMBEDDING: [
        "embedding vector representation features neural network output",
        "dense vector embedding space high dimensional float array",
    ],
    SemanticType.TARGET_LABEL: [
        "target label outcome prediction output class y dependent variable",
        "response variable ground truth supervised learning label",
    ],
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _build_column_text(col_name: str, dtype: pl.DataType, series: pl.Series) -> str:
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

        sample_value = str(non_null[0]) if len(non_null) > 0 else ""

    parts = [
        col_name.replace("_", " ").replace("-", " "),
        str(dtype),
        " ".join(top_values),
        sample_value,
    ]
    return " ".join(p for p in parts if p).strip()


def _encode(model, texts: list[str]) -> np.ndarray:
    return np.array(list(model.embed(texts)), dtype=np.float32)


class EmbeddingClassifier:
    def __init__(self) -> None:
        self._model = None
        self._anchor_embeddings: Optional[dict[SemanticType, np.ndarray]] = None

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
        self._model = TextEmbedding(model_name=_MODEL_NAME, show_progress=False)

    def _build_anchors(self):
        if self._anchor_embeddings is not None:
            return
        self._load_model()
        self._anchor_embeddings = {}
        for semantic_type, texts in _ANCHOR_TEXTS.items():
            embeddings = _encode(self._model, texts)
            self._anchor_embeddings[semantic_type] = embeddings.mean(axis=0)

    def classify_column(
        self,
        col_name: str,
        dtype: pl.DataType,
        series: pl.Series,
    ) -> ColumnClassification:
        self._build_anchors()

        col_text = _build_column_text(col_name, dtype, series)
        col_embedding = _encode(self._model, [col_text])[0]

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
        raw_confidence = 0.70 + (best_sim - 0.5) * 0.4 + margin * 0.2
        confidence = float(np.clip(raw_confidence, 0.70, 0.90))

        return ColumnClassification(
            name=col_name,
            semantic_type=best_type,
            confidence=confidence,
            reasoning=f"embedding similarity {best_sim:.3f} to {best_type.value} anchor (margin {margin:.3f})",
            source="embedding",
        )

    def classify_batch(
        self,
        columns: list[tuple[str, pl.DataType, pl.Series]],
    ) -> list[ColumnClassification]:
        self._build_anchors()

        texts = [_build_column_text(name, dtype, series) for name, dtype, series in columns]
        col_embeddings = _encode(self._model, texts)  # (N, D)

        results = []
        for i, (col_name, dtype, series) in enumerate(columns):
            col_embedding = col_embeddings[i]
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
            raw_confidence = 0.70 + (best_sim - 0.5) * 0.4 + margin * 0.2
            confidence = float(np.clip(raw_confidence, 0.70, 0.90))

            results.append(ColumnClassification(
                name=col_name,
                semantic_type=best_type,
                confidence=confidence,
                reasoning=f"embedding similarity {best_sim:.3f} to {best_type.value} anchor (margin {margin:.3f})",
                source="embedding",
            ))

        return results


_shared_classifier: Optional[EmbeddingClassifier] = None


def get_embedding_classifier() -> EmbeddingClassifier:
    global _shared_classifier
    if _shared_classifier is None:
        _shared_classifier = EmbeddingClassifier()
    return _shared_classifier