import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .writers import write


@dataclass
class WriteConfig:
    format: str
    path: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)
    write_mode: str = "replace"
    primary_keys: list[str] = field(default_factory=list)
    connector_config: dict[str, Any] | None = None


@dataclass
class ExportResult:
    run_id: str
    rows_written: int
    path: Path | None
    destination: str


class ExportPipeline:
    def __init__(self, config: WriteConfig) -> None:
        self.config = config
        self.run_id = uuid.uuid4().hex[:8]

    def run(self, df: pl.DataFrame, dry_run: bool = False) -> ExportResult:
        cfg = self.config

        if cfg.connector_config:
            return self._write_back(df, dry_run)

        if cfg.path is None:
            raise ValueError("Either path or connector_config must be provided")

        if dry_run:
            print(df.head(10))
            return ExportResult(
                run_id=self.run_id,
                rows_written=0,
                path=None,
                destination="dry_run",
            )

        write(df, cfg.path, cfg.format, **cfg.options)
        return ExportResult(
            run_id=self.run_id,
            rows_written=len(df),
            path=cfg.path,
            destination=str(cfg.path),
        )

    def _write_back(self, df: pl.DataFrame, dry_run: bool) -> ExportResult:
        import asyncio

        cfg = self.config
        source = cfg.connector_config.get("source", "")

        if dry_run:
            print(df.head(10))
            return ExportResult(self.run_id, 0, None, f"dry_run:{source}")

        if source == "postgresql":
            asyncio.run(_pg_write(df, cfg))
        elif source == "mysql":
            asyncio.run(_mysql_write(df, cfg))
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


async def _pg_write(df: pl.DataFrame, cfg: WriteConfig) -> None:
    import asyncpg

    cc = cfg.connector_config
    conn = await asyncpg.connect(
        host=cc["host"], port=cc.get("port", 5432),
        database=cc["database"], user=cc["user"], password=cc["password"],
    )
    table = cc["table"]

    try:
        if cfg.write_mode == "replace":
            await conn.execute(f'TRUNCATE TABLE "{table}"')
            await _pg_insert(conn, df, table)

        elif cfg.write_mode == "append":
            await _pg_insert(conn, df, table)

        elif cfg.write_mode == "upsert":
            if not cfg.primary_keys:
                raise ValueError("primary_keys required for upsert")
            await _pg_upsert(conn, df, table, cfg.primary_keys)

        else:
            raise ValueError(f"Unknown write_mode: {cfg.write_mode}")
    finally:
        await conn.close()


async def _pg_insert(conn, df: pl.DataFrame, table: str) -> None:
    cols = df.columns
    placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
    col_list = ", ".join(cols)
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
    rows = [tuple(row.values()) for row in df.to_dicts()]
    await conn.executemany(sql, rows)


async def _pg_upsert(conn, df: pl.DataFrame, table: str, primary_keys: list[str]) -> None:
    cols = df.columns
    placeholders = ", ".join(f"${i + 1}" for i in range(len(cols)))
    col_list = ", ".join(cols)
    conflict_cols = ", ".join(primary_keys)
    update_set = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in cols if c not in primary_keys
    )
    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_set}"
    )
    rows = [tuple(row.values()) for row in df.to_dicts()]
    await conn.executemany(sql, rows)


async def _mysql_write(df: pl.DataFrame, cfg: WriteConfig) -> None:
    import asyncmy

    cc = cfg.connector_config
    conn = await asyncmy.connect(
        host=cc["host"], port=cc.get("port", 3306),
        db=cc["database"], user=cc["user"], password=cc["password"],
    )
    table = cc["table"]
    cols = df.columns
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(cols)
    rows = [tuple(row.values()) for row in df.to_dicts()]

    try:
        async with conn.cursor() as cur:
            if cfg.write_mode == "replace":
                await cur.execute(f"TRUNCATE TABLE {table}")
                await cur.executemany(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})", rows
                )
            elif cfg.write_mode == "append":
                await cur.executemany(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})", rows
                )
            elif cfg.write_mode == "upsert":
                update_set = ", ".join(f"{c}=VALUES({c})" for c in cols if c not in cfg.primary_keys)
                await cur.executemany(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                    f"ON DUPLICATE KEY UPDATE {update_set}",
                    rows,
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