from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from dataprep.connectors.local_file import LocalFileConnector
from dataprep.core.context import PipelineContext
from dataprep.core.events import EventType
from dataprep.features.ingest.pipeline import IngestConfig, IngestPipeline
from dataprep.features.profile.pipeline import ProfileOptions, ProfilePipeline
from dataprep.storage.artifact import ArtifactStore

app = typer.Typer(name="dp", help="DataPrep CLI — data preprocessing framework", no_args_is_help=True)
console = Console()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_context(out: str) -> PipelineContext:
    return PipelineContext(artifact_store=ArtifactStore(out))


@app.command("ingest")
def cmd_ingest(
    source: str = typer.Option(..., "--source", "-s", help="Connector type: local_file | postgresql"),
    path: Optional[str] = typer.Option(None, "--path", help="File path (local_file connector)"),
    out: str = typer.Option("src/dataprep/artifacts", "--out", "-o", help="Artifact output directory"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max rows to read"),
    batch_size: int = typer.Option(50_000, "--batch-size", help="Rows per batch"),
):
    """Ingest data from a connector into artifact store."""
    async def _run_ingest():
        if source == "local_file":
            if not path:
                console.print("[red]--path is required for local_file connector[/red]")
                raise typer.Exit(1)
            connector = LocalFileConnector()
            query = {"path": path}
        elif source == "postgresql":
            console.print("[red]postgresql connector requires config file — not yet supported via CLI flags[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[red]Unknown source: {source}[/red]")
            raise typer.Exit(1)

        options: dict = {"batch_size": batch_size}
        if limit:
            options["n_rows"] = limit

        pipeline = IngestPipeline(IngestConfig(connector=connector, query=query, options=options))
        ctx = _make_context(out)

        result = pipeline.validate(ctx)
        if not result.ok:
            for err in result.errors:
                console.print(f"[red]Validation error: {err}[/red]")
            raise typer.Exit(1)

        plan = pipeline.plan(ctx)

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task("Ingesting...", total=None)
            async for event in pipeline.execute(plan, ctx):
                progress.update(task, description=event.message)
                if event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    console.print(f"\n[green]✓ {event.message}[/green]")
                    console.print(f"  Artifact: [cyan]{payload.get('output_artifact_id')}[/cyan]")
                    console.print(f"  Schema:   [cyan]{payload.get('schema_artifact_id')}[/cyan]")
                elif event.event_type == EventType.ERROR:
                    console.print(f"\n[red]✗ {event.message}[/red]")
                    raise typer.Exit(1)

    _run(_run_ingest())


@app.command("profile")
def cmd_profile(
    input: str = typer.Option(..., "--input", "-i", help="Artifact ID or Parquet file path"),
    mode: str = typer.Option("full", "--mode", "-m", help="full | summary"),
    sample_strategy: str = typer.Option("none", "--sample-strategy", help="none | random | reservoir"),
    sample_size: int = typer.Option(100_000, "--sample-size"),
    correlation: str = typer.Option("pearson", "--correlation", help="pearson | spearman | none"),
    out: str = typer.Option("src/dataprep/artifacts", "--out", "-o"),
):
    """Run profile pipeline on an ingested dataset."""
    async def _run_profile():
        pipeline = ProfilePipeline(ProfileOptions(
            mode=mode,
            sample_strategy=sample_strategy,
            sample_size=sample_size,
            correlation_method=correlation,
        ))
        ctx = _make_context(out)

        result = pipeline.validate(ctx)
        if not result.ok:
            for err in result.errors:
                console.print(f"[red]Validation error: {err}[/red]")
            raise typer.Exit(1)

        plan = pipeline.plan(ctx)
        if Path(input).exists() and Path(input).suffix == ".parquet":
            plan.metadata["dataframe"] = pl.read_parquet(input)
        else:
            plan.metadata["input_artifact_id"] = input

        with Progress(
            SpinnerColumn(), BarColumn(),
            TextColumn("{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Profiling...", total=100)
            async for event in pipeline.execute(plan, ctx):
                pct = int((event.progress_pct or 0) * 100)
                progress.update(task, completed=pct, description=event.message)
                if event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    console.print(f"\n[green]✓ {event.message}[/green]")
                    console.print(f"  Profile ID: [cyan]{payload.get('profile_id')}[/cyan]")
                    console.print(f"  Detail:     [cyan]{payload.get('detail_artifact_id')}[/cyan]")
                    console.print(f"  Summary:    [cyan]{payload.get('summary_artifact_id')}[/cyan]")
                    _print_profile_table(ctx, payload.get("summary_artifact_id"))
                elif event.event_type == EventType.ERROR:
                    console.print(f"\n[red]✗ {event.message}[/red]")
                    raise typer.Exit(1)

    _run(_run_profile())


@app.command("connector")
def cmd_connector(
    action: str = typer.Argument(..., help="test | info"),
    source: str = typer.Option(..., "--source", "-s"),
    path: Optional[str] = typer.Option(None, "--path"),
):
    """Test or inspect a connector."""
    async def _run_connector():
        if source == "local_file":
            connector = LocalFileConnector({"default_path": path or "."})
        else:
            console.print(f"[red]Unknown source: {source}[/red]")
            raise typer.Exit(1)

        if action == "test":
            status = await connector.test_connection()
            if status.ok:
                console.print(f"[green]✓ Connection OK ({status.latency_ms:.1f}ms)[/green]")
            else:
                console.print(f"[red]✗ Connection failed: {status.error}[/red]")
        elif action == "info" and path:
            schema = await connector.schema()
            table = Table(title=f"Schema: {path}")
            table.add_column("Column")
            table.add_column("Type")
            table.add_column("Nullable")
            for col in schema.columns:
                table.add_row(col.name, col.dtype, "yes" if col.nullable else "no")
            console.print(table)

    _run(_run_connector())


def _print_profile_table(ctx: PipelineContext, summary_id: str | None) -> None:
    if not summary_id:
        return
    try:
        summary = asyncio.get_event_loop().run_until_complete(
            ctx.artifact_store.load_json(summary_id)
        )
        table = Table(title="Column Summary", show_lines=True)
        table.add_column("Column", style="bold")
        table.add_column("Type")
        table.add_column("Null%", justify="right")
        table.add_column("Distinct", justify="right")
        table.add_column("Min")
        table.add_column("Max")
        table.add_column("Warnings", style="yellow")
        for col in summary.get("columns", []):
            table.add_row(
                col["name"],
                col["dtype"],
                f"{col['null_pct'] * 100:.1f}%",
                str(col["distinct_count"]),
                str(col.get("min", "")),
                str(col.get("max", "")),
                ", ".join(col.get("warnings", [])),
            )
        console.print(table)
    except Exception:
        pass


if __name__ == "__main__":
    app()