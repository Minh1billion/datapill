from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from datapill.core.events import EventType
from datapill.features.ingest.pipeline import IngestConfig, IngestPipeline

from ._shared import SOURCES, build_connector, console, load_config, make_context, run_async, validate_pipeline

app = typer.Typer()


@app.command("ingest")
def cmd_ingest(
    source: str = typer.Option(..., "--source", "-s", help=f"Connector type: {SOURCES}"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to JSON config file for connector"),
    path: Optional[str] = typer.Option(None, "--path", help="File path (local_file)"),
    table: Optional[str] = typer.Option(None, "--table", help="Table name (postgresql | mysql)"),
    url: Optional[str] = typer.Option(None, "--url", help="S3 URL, e.g. s3://bucket/key.parquet"),
    topic: Optional[str] = typer.Option(None, "--topic", help="Kafka topic name"),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", help="REST endpoint, e.g. /users"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max rows to read"),
    batch_size: int = typer.Option(50_000, "--batch-size", help="Rows per batch"),
    max_records: Optional[int] = typer.Option(None, "--max-records", help="Max records (kafka)"),
    no_materialize: bool = typer.Option(False, "--no-materialize", help="Skip Parquet write; store connector ref only. Source must remain available for downstream commands."),
):
    """Ingest data from a connector into artifact store."""
    async def _exec():
        config = load_config(config_file)
        connector, query = build_connector(source, config, path, table, url, topic, endpoint)

        options: dict = {"batch_size": batch_size}
        if limit:
            options["n_rows"] = limit
        if max_records:
            options["max_records"] = max_records

        pipeline = IngestPipeline(IngestConfig(
            connector=connector,
            query=query,
            options=options,
            materialize=not no_materialize,
        ))
        ctx = make_context()
        validate_pipeline(pipeline, ctx)
        plan = pipeline.plan(ctx)

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:
            task = p.add_task("Ingesting...", total=None)
            async for event in pipeline.execute(plan, ctx):
                p.update(task, description=event.message)
                if event.event_type == EventType.PROGRESS and event.message.startswith("Warning:"):
                    console.print(f"\n[yellow]{event.message}[/yellow]")
                elif event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    console.print(f"\n[green][OK] {event.message}[/green]")
                    console.print(f"  Artifact: [cyan]{payload.get('output_artifact_id')}[/cyan]")
                    console.print(f"  Schema:   [cyan]{payload.get('schema_artifact_id')}[/cyan]")
                    if payload.get("materialize"):
                        console.print(f"  Rows:     [cyan]{payload.get('rows_read', 0):,}[/cyan]")
                    else:
                        console.print("  [yellow]No data materialized - downstream commands will re-stream from source[/yellow]")
                elif event.event_type == EventType.ERROR:
                    console.print(f"\n[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

    run_async(_exec())