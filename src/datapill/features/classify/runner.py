import polars as pl

from .schema import ClassifyConfig, ClassifyResult, ColumnClassification, SemanticType, ProfileSignals
from .rule_based import classify_column_rule_based
from .embedding import get_embedding_classifier


def run_classify(
    df: pl.DataFrame,
    config: ClassifyConfig,
    profile_signals_map: dict[str, ProfileSignals] | None = None,
) -> ClassifyResult:
    if config.mode == "rule_based":
        raw = _rule_only(df, profile_signals_map)
    elif config.mode == "embedding":
        raw = _embedding_only(df, profile_signals_map, cache_dir=config.model_cache_dir)
    else:
        raw = _hybrid_classify(df, profile_signals_map, cache_dir=config.model_cache_dir)

    columns: list[ColumnClassification] = []
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

    return ClassifyResult(columns=columns, profile_used=profile_signals_map is not None)


def _rule_only(
    df: pl.DataFrame,
    profile_signals_map: dict[str, ProfileSignals] | None,
) -> list[ColumnClassification]:
    results = []
    for col in df.columns:
        signals = profile_signals_map.get(col) if profile_signals_map else None
        result = classify_column_rule_based(col, df[col].dtype, df[col], profile_signals=signals)
        if result is None:
            result = ColumnClassification(
                name=col,
                semantic_type=SemanticType.UNKNOWN,
                confidence=0.0,
                reasoning="no certain rule matched",
                source="rule_based",
            )
        results.append(result)
    return results


def _embedding_only(
    df: pl.DataFrame,
    profile_signals_map: dict[str, ProfileSignals] | None,
    cache_dir: str | None = None,
) -> list[ColumnClassification]:
    classifier = get_embedding_classifier(cache_dir=cache_dir)
    batch = [(col, df[col].dtype, df[col]) for col in df.columns]
    return classifier.classify_batch(batch, profile_signals_map=profile_signals_map)


def _hybrid_classify(
    df: pl.DataFrame,
    profile_signals_map: dict[str, ProfileSignals] | None,
    cache_dir: str | None = None,
) -> list[ColumnClassification]:
    classifier = get_embedding_classifier(cache_dir=cache_dir)
    emb_batch = [(col, df[col].dtype, df[col]) for col in df.columns]
    embedding_results = {
        r.name: r
        for r in classifier.classify_batch(emb_batch, profile_signals_map=profile_signals_map)
    }

    results = []
    for col in df.columns:
        signals = profile_signals_map.get(col) if profile_signals_map else None
        rule_result = classify_column_rule_based(col, df[col].dtype, df[col], profile_signals=signals)

        if rule_result is not None:
            results.append(rule_result)
        else:
            results.append(embedding_results[col])

    return results