import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from ..connectors import registry
from ..core.context import Context
from ..core.events import EventType
from ..features.ingest.pipeline import IngestPipeline
from ..storage.artifact_store import ArtifactStore
from .shared import (
    exit_on_error,
    print_artifact_path,
    print_connection_result,
    print_event,
    print_read_result,
    print_run_summary,
    print_schema,
)

app = typer.Typer(help="ingest data from a source into datapill")


def _load_config(value: str) -> dict:
    p = Path(value)
    if not p.exists():
        typer.echo(f"[fail] config file not found: {p}", err=True)
        raise typer.Exit(1)
    if p.suffix != ".json":
        typer.echo(f"[fail] config file must be a .json file: {p}", err=True)
        raise typer.Exit(1)
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        typer.echo(f"[fail] invalid config file {p}: {exc}", err=True)
        raise typer.Exit(1)


@app.command()
def run(
    source: str = typer.Argument(help="source type: postgres, mysql, sqlite, s3, local, kafka, rest"),
    config: str = typer.Option(..., "--config", "-c", help="path to config .json file"),
    table: Optional[str] = typer.Option(None, "--table", "-t", help="table name (postgres, mysql, sqlite)"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="raw SQL query (postgres, mysql, sqlite)"),
    topic: Optional[str] = typer.Option(None, "--topic", help="kafka topic"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="file path (s3, local)"),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", "-e", help="REST endpoint path"),
    params: Optional[str] = typer.Option(None, "--params", help="path to params .json file"),
    sample: bool = typer.Option(False, "--sample", help="read a sample instead of full data"),
    sample_size: int = typer.Option(10_000, "--sample-size", help="number of rows to sample"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="rows per streaming batch"),
    materialized: bool = typer.Option(False, "--materialize", "-m", help="write output to parquet artifact"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
    schema: bool = typer.Option(False, "--schema", help="print column schema after ingest"),
) -> None:
    connector_config = _load_config(config)

    options: dict = {}

    if table:
        options["table"] = table
    if query:
        options["query"] = query
    if topic:
        options["topic"] = topic
    if path:
        options["path"] = path
    if endpoint:
        options["endpoint"] = endpoint
    if params:
        options["params"] = _load_config(params)
    if batch_size:
        options["batch_size"] = batch_size

    options["sample"] = sample
    options["sample_size"] = sample_size
    options["materialized"] = materialized

    artifact_store = ArtifactStore(store_path)
    context = Context(artifact_store=artifact_store)
    pipeline = IngestPipeline(source=source, config=connector_config, options=options)

    validation = pipeline.validate(context)
    if not validation.ok:
        for err in validation.errors:
            typer.echo(f"[fail] {err}", err=True)
        raise typer.Exit(1)

    async def _run() -> None:
        plan = pipeline.plan(context)
        async for event in pipeline.execute(plan, context):
            if event.event_type == EventType.PROGRESS and event.payload:
                if "latency_ms" in event.payload:
                    print_connection_result(event.payload["latency_ms"])
                    continue
                if "rows" in event.payload and "columns" in event.payload:
                    print_read_result(event.payload["rows"], event.payload["columns"])
                    continue
            print_event(event)
            exit_on_error(event)

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
    config: str = typer.Option(..., "--config", "-c", help="path to config .json file"),
) -> None:
    connector_config = _load_config(config)

    async def _check() -> None:
        connector = registry.build(source, connector_config)
        status = await connector.connect()
        if status.ok:
            print_connection_result(status.latency_ms)
            print("ok")
        else:
            typer.echo(f"[fail] {status.error}", err=True)
            raise typer.Exit(1)
        await connector.cleanup()

    asyncio.run(_check())