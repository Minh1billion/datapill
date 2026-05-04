from typing import Any

import polars as pl

from ..connectors import registry
from ..core.context import Context
from ..storage.artifact_store import Artifact
from ..features.ingest.readers import get_reader


def _find_ingest_ancestor(artifact: Artifact, context: Context) -> Artifact:
    current = artifact
    seen: set[str] = set()
    while current.pipeline != "ingest":
        if current.run_id in seen:
            raise RuntimeError(f"circular artifact chain detected at {current.run_id!r}")
        seen.add(current.run_id)

        if not current.parent_run_id:
            raise RuntimeError(
                f"artifact {current.run_id!r} (pipeline={current.pipeline!r}) "
                "has no parent and is not an ingest artifact"
            )
        parent = context.artifact_store.get(current.parent_run_id)
        if parent is None:
            raise RuntimeError(
                f"parent artifact not found: {current.parent_run_id!r} "
                f"(referenced by {current.run_id!r})"
            )
        current = parent
    return current


async def load_dataframe(artifact: Artifact, context: Context) -> pl.DataFrame:
    # 1. Fast path: artifact already materialized as parquet
    if artifact.materialized and artifact.path:
        abs_path = context.artifact_store.path / artifact.path
        return pl.read_parquet(abs_path)

    # 2. If this is a preprocess (or any non-ingest) artifact without a parquet,
    #    trace the lineage chain upward until we find the ingest artifact.
    ingest_artifact = _find_ingest_ancestor(artifact, context)

    # 3. If the ingest artifact itself is materialized, read its parquet directly
    if ingest_artifact.materialized and ingest_artifact.path:
        abs_path = context.artifact_store.path / ingest_artifact.path
        return pl.read_parquet(abs_path)

    # 4. Otherwise re-run the connector to load data from source
    source = ingest_artifact.options.get("source")
    if not source:
        raise RuntimeError(
            f"ingest artifact {ingest_artifact.run_id!r} has no 'source' in options"
        )

    config = ingest_artifact.options.get("connector_config", {})
    connector = registry.build(source, config)
    status = await connector.connect()
    if not status.ok:
        raise RuntimeError(f"connection failed: {status.error}")

    try:
        reader = get_reader(source)
        stream = await reader.read(
            connector=connector,
            options=ingest_artifact.options,
            is_sample=ingest_artifact.is_sample,
            sample_size=ingest_artifact.sample_size or 10_000,
        )
        batches: list[pl.DataFrame] = []
        async for chunk in stream:
            batches.append(chunk)
    finally:
        await connector.cleanup()

    return pl.concat(batches) if batches else pl.DataFrame()