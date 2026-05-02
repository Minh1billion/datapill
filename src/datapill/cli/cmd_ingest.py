import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from ..connectors import registry
from ..core.context import Context
from ..features.ingest.pipeline import IngestPipeline
from ..storage.artifact_store import ArtifactStore
from .shared import (
    print_artifact_path,
    print_connection_result,
    print_run_summary,
    print_schema,
    run_pipeline,
)

app = typer.Typer(help="ingest data from a source into datapill")

_CONFIG_REQUIRED_SOURCES = {"postgres", "mysql", "sqlite", "kafka", "rest", "s3"}


def _load_config(value: str) -> dict:
    p = Path(value)
    if not p.exists():
        typer.echo(f"[FAIL] config file not found: {p}", err=True)
        raise typer.Exit(1)
    if p.suffix != ".json":
        typer.echo(f"[FAIL] config file must be a .json file: {p}", err=True)
        raise typer.Exit(1)
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        typer.echo(f"[FAIL] invalid config file {p}: {exc}", err=True)
        raise typer.Exit(1)


def _resolve_local_path(path: str, config: dict) -> tuple[str, str]:
    p = Path(path)

    if "base_path" in config:
        base = Path(config["base_path"])
        if p.is_absolute():
            try:
                rel = p.relative_to(base)
            except ValueError:
                typer.echo(
                    f"[FAIL] --path {path!r} is not inside base_path {str(base)!r} from config",
                    err=True,
                )
                raise typer.Exit(1)
            return str(base), str(rel)
        if not (base / p).exists():
            typer.echo(f"[FAIL] file not found: {base / p}", err=True)
            raise typer.Exit(1)
        return str(base), str(p)

    p = p.resolve()
    if not p.exists():
        typer.echo(f"[FAIL] file not found: {p}", err=True)
        raise typer.Exit(1)
    if p.is_dir():
        typer.echo(f"[FAIL] --path must point to a file, not a directory: {p}", err=True)
        raise typer.Exit(1)
    return str(p.parent), p.name


@app.command()
def run(
    source: str = typer.Argument(help="source type: postgres, mysql, sqlite, s3, local, kafka, rest"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="path to config .json file (required for postgres, mysql, sqlite, kafka, rest, s3; optional for local)"),
    table: Optional[str] = typer.Option(None, "--table", "-t", help="table name (postgres, mysql, sqlite)"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="raw SQL query (postgres, mysql, sqlite)"),
    topic: Optional[str] = typer.Option(None, "--topic", help="kafka topic"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="file path (local: full path to file; s3: key within bucket)"),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", "-e", help="REST endpoint path"),
    params: Optional[str] = typer.Option(None, "--params", help="path to params .json file"),
    sample: bool = typer.Option(False, "--sample", help="read a sample instead of full data"),
    sample_size: int = typer.Option(10_000, "--sample-size", help="number of rows to sample"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="rows per streaming batch"),
    materialized: bool = typer.Option(False, "--materialize", "-m", help="write output to parquet artifact"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
    schema: bool = typer.Option(False, "--schema", help="print column schema after ingest"),
    mkdir: bool = typer.Option(False, "--mkdir", help="create base_path directory if it does not exist (local source only)"),
) -> None:
    if source in _CONFIG_REQUIRED_SOURCES and not config:
        typer.echo(f"[FAIL] --config is required for source '{source}'", err=True)
        raise typer.Exit(1)

    connector_config = _load_config(config) if config else {}

    if table and query:
        typer.echo("[FAIL] cannot use both --table and --query", err=True)
        raise typer.Exit(1)

    options: dict = {}

    if query:
        options["query"] = query
    elif table:
        options["query"] = f"SELECT * FROM {table}"
    if topic:
        options["topic"] = topic
    if endpoint:
        options["endpoint"] = endpoint
    if params:
        options["params"] = _load_config(params)
    if batch_size:
        options["batch_size"] = batch_size

    if source == "local":
        if not path:
            typer.echo("[FAIL] --path is required for source 'local'", err=True)
            raise typer.Exit(1)
        base_path, rel_path = _resolve_local_path(path, connector_config)
        connector_config = {**connector_config, "base_path": base_path, "mkdir": mkdir}
        options["path"] = rel_path
    elif path:
        options["path"] = path

    options["source"] = source
    options["sample"] = sample
    options["sample_size"] = sample_size
    options["materialized"] = materialized

    artifact_store = ArtifactStore(store_path)
    context = Context(artifact_store=artifact_store)
    pipeline = IngestPipeline(source=source, config=connector_config, options=options)

    validation = pipeline.validate(context)
    if not validation.ok:
        for e in validation.errors:
            typer.echo(f"[FAIL] {e}", err=True)
        raise typer.Exit(1)

    async def _run() -> None:
        plan = pipeline.plan(context)
        await run_pipeline(pipeline.execute(plan, context))

        if context.artifact:
            art = context.artifact
            print_run_summary({"run_id": art.run_id, "ref": art.ref})
            if schema and art.schema:
                print_schema(art.schema)
            if art.materialized and art.path:
                print_artifact_path(art.path)

    asyncio.run(_run())


@app.command("sources")
def list_sources() -> None:
    for s in registry.sources():
        print(s)


@app.command("check")
def check_connection(
    source: str = typer.Argument(help="source type"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="path to config .json file"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="file path (for local source)"),
    mkdir: bool = typer.Option(False, "--mkdir", help="create base_path directory if it does not exist (local source only)"),
) -> None:
    if source in _CONFIG_REQUIRED_SOURCES and not config:
        typer.echo(f"[FAIL] --config is required for source '{source}'", err=True)
        raise typer.Exit(1)

    connector_config = _load_config(config) if config else {}

    if source == "local":
        if not path and "base_path" not in connector_config:
            typer.echo("[FAIL] --path or 'base_path' in config required for source 'local'", err=True)
            raise typer.Exit(1)
        if path:
            base_path, _ = _resolve_local_path(path, connector_config)
            connector_config = {**connector_config, "base_path": base_path, "mkdir": mkdir}

    async def _check() -> None:
        connector = registry.build(source, connector_config)
        status = await connector.connect()
        if status.ok:
            print_connection_result(status.latency_ms)
        else:
            typer.echo(f"[FAIL] {status.error}", err=True)
            raise typer.Exit(1)
        await connector.cleanup()

    asyncio.run(_check())