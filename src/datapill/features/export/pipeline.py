import asyncio
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Any

import polars as pl

from datapill.core.context import PipelineContext
from datapill.core.events import EventType, ProgressEvent
from datapill.core.interfaces import ExecutionPlan, FeaturePipeline, ValidationResult
from .schema import WriteConfig, ExportResult
from .writers import write

_DEFAULT_BATCH = 5_000

_KAFKA_RESTREAM_WARNING = (
    "Warning: input is a Kafka ref - re-consuming from topic. "
    "Offsets will advance and data may differ from original ingest."
)
_RESTREAM_WARNING = (
    "Warning: input is a ref artifact - re-streaming from source. "
    "Source must remain available."
)


class ExportPipeline(FeaturePipeline):
    def __init__(self, config: WriteConfig) -> None:
        self.config = config
        self.run_id = uuid.uuid4().hex[:8]

    def validate(self, context: PipelineContext) -> ValidationResult:
        errors: list[str] = []
        if not self.config.format:
            errors.append("format must not be empty")
        if self.config.connector_config is None and self.config.path is None:
            errors.append("Either path or connector_config must be provided")
        if self.config.write_mode not in ("replace", "append", "upsert"):
            errors.append(f"Invalid write_mode: {self.config.write_mode}")
        if self.config.write_mode == "upsert" and not self.config.primary_keys:
            errors.append("primary_keys required for upsert write_mode")
        return ValidationResult(ok=len(errors) == 0, errors=errors)

    def plan(self, context: PipelineContext) -> ExecutionPlan:
        steps = (
            ["stream_write_back"] if self.config.connector_config else ["write_file"]
        )
        destination = (
            self.config.connector_config.get("source", "")
            if self.config.connector_config
            else str(self.config.path)
        )
        return ExecutionPlan(
            steps=[{"name": s} for s in steps],
            metadata={
                "format": self.config.format,
                "write_mode": self.config.write_mode,
                "destination": destination,
            },
        )

    async def execute(
        self, plan: ExecutionPlan, context: PipelineContext
    ) -> AsyncGenerator[ProgressEvent, None]:
        t0 = time.perf_counter()
        dry_run: bool = plan.metadata.get("dry_run", False)

        yield ProgressEvent(EventType.STARTED, "Export started", progress_pct=0.0)

        input_ref = plan.metadata.get("input_artifact_id")
        df: pl.DataFrame | None = None

        if input_ref and context.artifact_store.is_ref(input_ref):
            ref = await context.artifact_store.load_ref(input_ref)
            from datapill.cli._shared import rebuild_connector_from_ref
            connector, query = rebuild_connector_from_ref(ref)

            if "Kafka" in ref.get("source_type", ""):
                yield ProgressEvent(EventType.PROGRESS, _KAFKA_RESTREAM_WARNING, progress_pct=0.02)
            else:
                yield ProgressEvent(EventType.PROGRESS, _RESTREAM_WARNING, progress_pct=0.02)

            chunks: list[pl.DataFrame] = []
            async for chunk in connector.read_stream(query, ref.get("options", {})):
                if not chunk.is_empty():
                    chunks.append(chunk)
            df = pl.concat(chunks) if chunks else pl.DataFrame()
            await connector.close()

        elif input_ref:
            df = context.artifact_store.scan_parquet(input_ref).collect()
        elif "dataframe" in plan.metadata:
            df = plan.metadata["dataframe"]
        else:
            yield ProgressEvent(EventType.ERROR, "No input data provided")
            return

        yield ProgressEvent(
            EventType.PROGRESS,
            f"Loaded {len(df):,} rows, writing to {plan.metadata['destination']}",
            progress_pct=0.1,
        )

        try:
            result = await self.run(df, dry_run=dry_run)
        except Exception as exc:
            yield ProgressEvent(EventType.ERROR, str(exc))
            raise

        duration = time.perf_counter() - t0
        yield ProgressEvent(
            EventType.DONE,
            f"Export complete in {duration:.1f}s - {result.rows_written:,} rows → {result.destination}",
            progress_pct=1.0,
            payload={
                "run_id": result.run_id,
                "rows_written": result.rows_written,
                "destination": result.destination,
                "duration_s": round(duration, 3),
            },
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "feature": "export",
            "format": self.config.format,
            "write_mode": self.config.write_mode,
            "path": str(self.config.path) if self.config.path else None,
            "connector_source": (
                self.config.connector_config.get("source")
                if self.config.connector_config
                else None
            ),
        }

    async def run(self, df: pl.DataFrame, dry_run: bool = False) -> ExportResult:
        cfg = self.config

        if cfg.connector_config:
            return await self._write_back(df, dry_run)

        if cfg.path is None:
            raise ValueError("Either path or connector_config must be provided")

        if dry_run:
            print(df.head(10))
            return ExportResult(run_id=self.run_id, rows_written=0, path=None, destination="dry_run")

        write(df, cfg.path, cfg.format, **cfg.options)
        return ExportResult(
            run_id=self.run_id,
            rows_written=len(df),
            path=cfg.path,
            destination=str(cfg.path),
        )

    async def _write_back(self, df: pl.DataFrame, dry_run: bool) -> ExportResult:
        cfg = self.config
        source = cfg.connector_config.get("source", "")

        if dry_run:
            print(df.head(10))
            return ExportResult(self.run_id, 0, None, f"dry_run:{source}")

        if source == "postgresql":
            await _pg_write(df, cfg)
        elif source == "mysql":
            await _mysql_write(df, cfg)
        elif source == "s3":
            _s3_write(df, cfg)
        else:
            raise ValueError(f"Write-back not supported for source: {source}")

        return ExportResult(
            run_id=self.run_id,
            rows_written=len(df),
            path=None,
            destination=source,
        )


def _iter_batches(rows: list, batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i: i + batch_size]


async def _pg_write(df: pl.DataFrame, cfg: WriteConfig) -> None:
    import asyncpg

    cc = cfg.connector_config
    batch_size: int = cfg.options.get("batch_size", _DEFAULT_BATCH)
    conn = await asyncpg.connect(
        host=cc["host"], port=cc.get("port", 5432),
        database=cc["database"], user=cc["user"], password=cc["password"],
    )
    table = cc["table"]

    try:
        if cfg.write_mode == "replace":
            await conn.execute(f'TRUNCATE TABLE "{table}"')
            await _pg_insert(conn, df, table, batch_size)
        elif cfg.write_mode == "append":
            await _pg_insert(conn, df, table, batch_size)
        elif cfg.write_mode == "upsert":
            if not cfg.primary_keys:
                raise ValueError("primary_keys required for upsert write_mode")
            await _pg_upsert(conn, df, table, cfg.primary_keys, batch_size)
        else:
            raise ValueError(f"Unknown write_mode: {cfg.write_mode}")
    finally:
        await conn.close()


async def _pg_insert(conn, df: pl.DataFrame, table: str, batch_size: int) -> None:
    cols = df.columns
    placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
    col_list = ", ".join(f'"{c}"' for c in cols)
    sql = f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})'

    rows = [tuple(row) for row in df.iter_rows()]
    try:
        for batch in _iter_batches(rows, batch_size):
            await conn.executemany(sql, batch)
    except Exception as exc:
        msg = str(exc)
        if "duplicate key" in msg or "unique constraint" in msg:
            raise ValueError(
                f"append failed: duplicate keys detected in table '{table}'. "
                f"Use --write-mode upsert --primary-keys <col> to handle conflicts, "
                f"or --write-mode replace to overwrite.\n"
                f"Detail: {msg}"
            ) from exc
        raise


async def _pg_upsert(conn, df: pl.DataFrame, table: str, primary_keys: list[str], batch_size: int) -> None:
    cols = df.columns
    placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
    col_list = ", ".join(f'"{c}"' for c in cols)
    conflict_cols = ", ".join(f'"{c}"' for c in primary_keys)
    update_set = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c not in primary_keys)
    sql = (
        f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders}) '
        f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_set}"
    )
    rows = [tuple(row) for row in df.iter_rows()]
    for batch in _iter_batches(rows, batch_size):
        await conn.executemany(sql, batch)


async def _mysql_write(df: pl.DataFrame, cfg: WriteConfig) -> None:
    import asyncmy

    cc = cfg.connector_config
    batch_size: int = cfg.options.get("batch_size", _DEFAULT_BATCH)
    conn = await asyncmy.connect(
        host=cc["host"], port=cc.get("port", 3306),
        db=cc["database"], user=cc["user"], password=cc["password"],
    )
    table = cc["table"]
    cols = df.columns
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(f"`{c}`" for c in cols)
    rows = [tuple(row) for row in df.iter_rows()]

    try:
        async with conn.cursor() as cur:
            if cfg.write_mode == "replace":
                await cur.execute(f"TRUNCATE TABLE `{table}`")
                for batch in _iter_batches(rows, batch_size):
                    await cur.executemany(
                        f"INSERT INTO `{table}` ({col_list}) VALUES ({placeholders})", batch
                    )
            elif cfg.write_mode == "append":
                try:
                    for batch in _iter_batches(rows, batch_size):
                        await cur.executemany(
                            f"INSERT INTO `{table}` ({col_list}) VALUES ({placeholders})", batch
                        )
                except Exception as exc:
                    msg = str(exc)
                    if "Duplicate entry" in msg or "duplicate" in msg.lower():
                        raise ValueError(
                            f"append failed: duplicate keys detected in table '{table}'. "
                            f"Use --write-mode upsert --primary-keys <col> to handle conflicts, "
                            f"or --write-mode replace to overwrite.\n"
                            f"Detail: {msg}"
                        ) from exc
                    raise
            elif cfg.write_mode == "upsert":
                update_set = ", ".join(
                    f"`{c}`=VALUES(`{c}`)" for c in cols if c not in cfg.primary_keys
                )
                for batch in _iter_batches(rows, batch_size):
                    await cur.executemany(
                        f"INSERT INTO `{table}` ({col_list}) VALUES ({placeholders}) "
                        f"ON DUPLICATE KEY UPDATE {update_set}",
                        batch,
                    )
            else:
                raise ValueError(f"Unknown write_mode: {cfg.write_mode}")
        await conn.commit()
    finally:
        conn.close()


def _s3_write(df: pl.DataFrame, cfg: WriteConfig) -> None:
    import io
    import boto3

    cc = cfg.connector_config
    s3 = boto3.client(
        "s3",
        aws_access_key_id=cc.get("aws_access_key_id"),
        aws_secret_access_key=cc.get("aws_secret_access_key"),
        region_name=cc.get("region", "us-east-1"),
        endpoint_url=cc.get("endpoint_url"),
    )
    key = cc["key"]
    fmt = cfg.format.lower()
    buf = io.BytesIO()

    if fmt == "parquet":
        df.write_parquet(buf, compression=cfg.options.get("compression", "snappy"))
    elif fmt == "csv":
        df.write_csv(buf)
    elif fmt == "json":
        df.write_json(buf)
    elif fmt == "jsonl":
        df.write_ndjson(buf)
    else:
        raise ValueError(f"S3 write: unsupported format '{fmt}'")

    buf.seek(0)
    s3.upload_fileobj(buf, cc["bucket"], key)