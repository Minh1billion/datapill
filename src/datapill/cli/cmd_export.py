from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from datapill.core.events import EventType
from datapill.features.export.pipeline import ExportPipeline
from datapill.features.export.schema import WriteConfig

from ._shared import FORMATS, console, load_config, make_context, resolve_input, run_async, validate_pipeline

app = typer.Typer()


@app.command("export")
def cmd_export(
    input: str = typer.Option(..., "--input", "-i", help="run_id or full artifact ID"),
    format: str = typer.Option(..., "--format", "-f", help=f"Output format: {FORMATS}"),
    out_path: Optional[str] = typer.Option(None, "--out-path", help="Output file path"),
    write_mode: str = typer.Option("replace", "--write-mode", help="replace | append | upsert"),
    primary_keys: Optional[str] = typer.Option(None, "--primary-keys", help="Comma-separated keys for upsert"),
    connector_file: Optional[str] = typer.Option(None, "--connector", "-c", help="Connector config JSON for write-back"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print first 10 rows, skip write"),
    compression: Optional[str] = typer.Option(None, "--compression", help="snappy | zstd | gzip (parquet only)"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Artifact store directory"),
):
    """Export a dataset to file or write back to a connector.

    Export to file:
        dp export -i <run_id> -f parquet --out-path output/result.parquet

    Write back to PostgreSQL:
        dp export -i <run_id> -f parquet --connector pg.json --write-mode upsert --primary-keys id
    """
    async def _exec():
        connector_config: dict | None = None
        if connector_file:
            connector_config = load_config(connector_file)
            if not connector_config.get("source"):
                console.print("[red]connector config must have 'source' field (postgresql | mysql | s3)[/red]")
                raise typer.Exit(1)

        if connector_config is None and not out_path:
            console.print("[red]--out-path is required when not using --connector[/red]")
            raise typer.Exit(1)

        options: dict = {}
        if compression:
            options["compression"] = compression

        pipeline = ExportPipeline(WriteConfig(
            format=format,
            path=Path(out_path) if out_path else None,
            options=options,
            write_mode=write_mode,
            primary_keys=[k.strip() for k in primary_keys.split(",")] if primary_keys else [],
            connector_config=connector_config,
        ))
        ctx = make_context(out)
        validate_pipeline(pipeline, ctx)
        plan = pipeline.plan(ctx)
        plan.metadata["dry_run"] = dry_run
        plan.metadata["input_artifact_id"] = resolve_input(ctx, input, feature_hint="export")

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:
            task = p.add_task("Exporting...", total=None)
            async for event in pipeline.execute(plan, ctx):
                p.update(task, description=event.message)
                if event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    console.print(f"\n[green][OK] {event.message}[/green]")
                    if dry_run:
                        console.print("  [yellow]Dry run - no data written[/yellow]")
                    else:
                        console.print(f"  Destination: [cyan]{payload.get('destination')}[/cyan]")
                        console.print(f"  Rows:        [cyan]{payload.get('rows_written', 0):,}[/cyan]")
                elif event.event_type == EventType.ERROR:
                    console.print(f"\n[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

    run_async(_exec())