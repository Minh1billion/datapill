from typing import Any

import polars as pl

from ...connectors import registry
from ...core.context import Context
from ...storage.artifact_store import Artifact
from ..ingest.readers import get_reader


async def load_dataframe(artifact: Artifact, context: Context) -> pl.DataFrame:
    if artifact.materialized and artifact.path:
        abs_path = context.artifact_store.path / artifact.path
        return pl.read_parquet(abs_path)

    source = artifact.options["source"]
    config = artifact.options.get("connector_config", {})
    connector = registry.build(source, config)
    status = await connector.connect()
    if not status.ok:
        raise RuntimeError(f"connection failed: {status.error}")

    try:
        reader = get_reader(source)
        stream = await reader.read(
            connector=connector,
            options=artifact.options,
            is_sample=artifact.is_sample,
            sample_size=artifact.sample_size or 10_000,
        )
        batches: list[pl.DataFrame] = []
        async for chunk in stream:
            batches.append(chunk)
    finally:
        await connector.cleanup()

    return pl.concat(batches) if batches else pl.DataFrame()