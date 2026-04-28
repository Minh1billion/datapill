import polars as pl

from .schema import ClassifyConfig, ClassifyResult, ColumnClassification, SemanticType
from .rule_based import classify_column_rule_based
from .embedding import get_embedding_classifier


def run_classify(df: pl.DataFrame, config: ClassifyConfig) -> ClassifyResult:
    columns: list[ColumnClassification] = []

    if config.mode == "embedding":
        classifier = get_embedding_classifier()
        batch = [(col, df[col].dtype, df[col]) for col in df.columns]
        raw = classifier.classify_batch(batch)
    elif config.mode == "rule_based":
        raw = [
            classify_column_rule_based(col, df[col].dtype, df[col])
            for col in df.columns
        ]
    else:
        raw = _hybrid_classify(df)

    for classification in raw:
        col_name = classification.name

        if col_name in config.overrides:
            try:
                overridden_type = SemanticType(config.overrides[col_name])
            except ValueError:
                overridden_type = SemanticType.UNKNOWN
            classification = ColumnClassification(
                name=col_name,
                semantic_type=overridden_type,
                confidence=1.0,
                reasoning=f"manually overridden to {overridden_type.value}",
                source="override",
                overridden=True,
            )

        if classification.confidence >= config.confidence_threshold:
            columns.append(classification)
        else:
            columns.append(ColumnClassification(
                name=col_name,
                semantic_type=SemanticType.UNKNOWN,
                confidence=classification.confidence,
                reasoning=f"confidence {classification.confidence:.3f} below threshold {config.confidence_threshold:.3f}",
                source=classification.source,
                overridden=False,
            ))

    return ClassifyResult(columns=columns)


def _hybrid_classify(df: pl.DataFrame) -> list[ColumnClassification]:
    rule_results = {
        col: classify_column_rule_based(col, df[col].dtype, df[col])
        for col in df.columns
    }

    ambiguous_cols = [
        col for col, r in rule_results.items()
        if r.semantic_type == SemanticType.UNKNOWN or r.confidence < 0.65
    ]

    embedding_results: dict[str, ColumnClassification] = {}
    if ambiguous_cols:
        classifier = get_embedding_classifier()
        batch = [(col, df[col].dtype, df[col]) for col in ambiguous_cols]
        for result in classifier.classify_batch(batch):
            embedding_results[result.name] = result

    final: list[ColumnClassification] = []
    for col in df.columns:
        rule_r = rule_results[col]
        if col in embedding_results:
            emb_r = embedding_results[col]
            if rule_r.semantic_type == SemanticType.UNKNOWN:
                final.append(emb_r)
            elif emb_r.confidence > rule_r.confidence:
                final.append(emb_r)
            else:
                final.append(rule_r)
        else:
            final.append(rule_r)

    return final