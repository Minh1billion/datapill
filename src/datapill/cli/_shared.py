import asyncio
import json
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich.console import Console

from datapill.connectors.base import BaseConnector
from datapill.connectors.kafka import KafkaConnector
from datapill.connectors.local_file import LocalFileConnector
from datapill.connectors.mysql import MySQLConnector
from datapill.connectors.postgresql import PostgreSQLConnector
from datapill.connectors.rest import RESTConnector
from datapill.connectors.s3 import S3Connector
from datapill.core.context import PipelineContext
from datapill.storage.artifact import ArtifactStore

console = Console()

SOURCES = "local_file | postgresql | mysql | s3 | rest | kafka"
FORMATS = "csv | parquet | json | jsonl | excel"


def run_async(coro):
    return asyncio.run(coro)


def make_context(out: str) -> PipelineContext:
    return PipelineContext(artifact_store=ArtifactStore(out))


def load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise typer.Exit(1)
    with p.open() as f:
        return json.load(f)


def require_db_config(config: dict, source: str) -> None:
    required = ["host", "database", "user", "password"]
    missing = [k for k in required if k not in config]
    if missing:
        console.print(f"[red]{source} config missing keys: {missing}[/red]")
        raise typer.Exit(1)


def build_connector(
    source: str,
    config: dict,
    path: str | None = None,
    table: str | None = None,
    url: str | None = None,
    topic: str | None = None,
    endpoint: str | None = None,
) -> tuple[BaseConnector, dict]:
    if source == "local_file":
        if not path:
            console.print("[red]--path is required for local_file[/red]")
            raise typer.Exit(1)
        return LocalFileConnector(config), {"path": path}

    if source == "postgresql":
        require_db_config(config, source)
        query: dict = {}
        if table:
            query["table"] = table
        elif "sql" in config:
            query["sql"] = config.pop("sql")
        else:
            console.print("[red]--table or 'sql' in config is required for postgresql[/red]")
            raise typer.Exit(1)
        return PostgreSQLConnector(config), query

    if source == "mysql":
        require_db_config(config, source)
        query = {}
        if table:
            query["table"] = table
        elif "sql" in config:
            query["sql"] = config.pop("sql")
        else:
            console.print("[red]--table or 'sql' in config is required for mysql[/red]")
            raise typer.Exit(1)
        return MySQLConnector(config), query

    if source == "s3":
        if not url and not config.get("default_url"):
            console.print("[red]--url or 'default_url' in config is required for s3[/red]")
            raise typer.Exit(1)
        return S3Connector(config), {"url": url or config["default_url"]}

    if source == "rest":
        if not endpoint and not config.get("default_endpoint"):
            console.print("[red]--endpoint or 'default_endpoint' in config is required for rest[/red]")
            raise typer.Exit(1)
        if "base_url" not in config:
            console.print("[red]'base_url' is required in config for rest[/red]")
            raise typer.Exit(1)
        return RESTConnector(config), {"endpoint": endpoint or config["default_endpoint"]}

    if source == "kafka":
        if not topic and not config.get("default_topic"):
            console.print("[red]--topic or 'default_topic' in config is required for kafka[/red]")
            raise typer.Exit(1)
        if "bootstrap_servers" not in config:
            console.print("[red]'bootstrap_servers' is required in config for kafka[/red]")
            raise typer.Exit(1)
        return KafkaConnector(config), {"topic": topic or config["default_topic"]}

    console.print(f"[red]Unknown source: {source}. Available: {SOURCES}[/red]")
    raise typer.Exit(1)


def resolve_input(ctx: PipelineContext, input_str: str, feature_hint: str) -> str:
    try:
        return ctx.artifact_store.resolve(input_str, feature_hint=feature_hint)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


def validate_pipeline(pipeline, ctx: PipelineContext) -> None:
    result = pipeline.validate(ctx)
    if not result.ok:
        for err in result.errors:
            console.print(f"[red]Validation error: {err}[/red]")
        raise typer.Exit(1)


def read_local(path: Path, fmt: str) -> pl.DataFrame:
    fmt = fmt.lower()
    if fmt == "csv":
        return pl.read_csv(path)
    if fmt == "parquet":
        return pl.read_parquet(path)
    if fmt == "json":
        return pl.read_json(path)
    if fmt in ("jsonl", "ndjson"):
        return pl.read_ndjson(path)
    if fmt in ("xlsx", "excel"):
        return pl.read_excel(path)
    raise ValueError(f"Unsupported format for read: '{fmt}'")


def write_local(df: pl.DataFrame, path: Path, fmt: str) -> None:
    fmt = fmt.lower()
    if fmt == "csv":
        df.write_csv(path)
    elif fmt == "parquet":
        df.write_parquet(path)
    elif fmt == "json":
        df.write_json(path)
    elif fmt in ("jsonl", "ndjson"):
        df.write_ndjson(path)
    elif fmt in ("xlsx", "excel"):
        df.write_excel(path)
    else:
        raise ValueError(f"Unsupported format for write: '{fmt}'")


async def db_exec(source: str, config: dict, sql: str) -> None:
    if source == "postgresql":
        import asyncpg
        conn = await asyncpg.connect(
            host=config["host"], port=config.get("port", 5432),
            database=config["database"], user=config["user"], password=config["password"],
        )
        try:
            result = await conn.execute(sql)
            console.print(f"[green][OK] {result}[/green]")
        finally:
            await conn.close()

    elif source == "mysql":
        import asyncmy
        import asyncmy.cursors
        conn = await asyncmy.connect(
            host=config["host"], port=config.get("port", 3306),
            db=config["database"], user=config["user"], password=config["password"],
        )
        try:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
                await cur.execute(sql)
                affected = cur.rowcount
            await conn.commit()
            console.print(f"[green][OK] {affected} row(s) affected[/green]")
        finally:
            conn.close()