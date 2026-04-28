import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from datapill.core.events import EventType
from datapill.features.profile.pipeline import ProfileOptions, ProfilePipeline

from ._shared import console, make_context, resolve_input, run_async, validate_pipeline
from ._tables import print_profile_table

app = typer.Typer()


@app.command("profile")
def cmd_profile(
    input: str = typer.Option(..., "--input", "-i", help="run_id or full artifact ID"),
    mode: str = typer.Option("full", "--mode", "-m", help="full | summary"),
    sample_strategy: str = typer.Option("none", "--sample-strategy", help="none | random | reservoir"),
    sample_size: int = typer.Option(100_000, "--sample-size"),
    correlation: str = typer.Option("pearson", "--correlation", help="pearson | spearman | none"),
    out: str = typer.Option("src/datapill/artifacts", "--out", "-o"),
):
    """Run profile pipeline on an ingested dataset."""
    async def _exec():
        pipeline = ProfilePipeline(ProfileOptions(
            mode=mode,
            sample_strategy=sample_strategy,
            sample_size=sample_size,
            correlation_method=correlation,
        ))
        ctx = make_context(out)
        validate_pipeline(pipeline, ctx)
        plan = pipeline.plan(ctx)
        plan.metadata["input_artifact_id"] = resolve_input(ctx, input, feature_hint="profile")

        with Progress(
            SpinnerColumn(), BarColumn(),
            TextColumn("{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as p:
            task = p.add_task("Profiling...", total=100)
            async for event in pipeline.execute(plan, ctx):
                p.update(task, completed=int((event.progress_pct or 0) * 100), description=event.message)
                if event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    console.print(f"\n[green][OK] {event.message}[/green]")
                    console.print(f"  Profile ID: [cyan]{payload.get('profile_id')}[/cyan]")
                    console.print(f"  Detail:     [cyan]{payload.get('detail_artifact_id')}[/cyan]")
                    console.print(f"  Summary:    [cyan]{payload.get('summary_artifact_id')}[/cyan]")
                    await print_profile_table(ctx, payload.get("summary_artifact_id"))
                elif event.event_type == EventType.ERROR:
                    console.print(f"\n[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

    run_async(_exec())