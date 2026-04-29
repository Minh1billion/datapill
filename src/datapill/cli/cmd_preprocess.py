import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from datapill.core.events import EventType
from datapill.features.preprocess.pipeline import PreprocessPipeline
from datapill.features.preprocess.schema import StepConfig

from ._shared import console, load_config, make_context, resolve_input, run_async, validate_pipeline

app = typer.Typer()


@app.command("preprocess")
def cmd_preprocess(
    input: str = typer.Option(..., "--input", "-i", help="run_id or full artifact ID"),
    pipeline_file: str = typer.Option(..., "--pipeline", "-p", help="Path to pipeline JSON config file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run on first 1000 rows, no artifact saved"),
    checkpoint: bool = typer.Option(False, "--checkpoint", help="Save checkpoint parquet after each step"),
):
    """Apply a preprocess pipeline to an ingested dataset.

    Pipeline JSON format:
    {
        "steps": [
            {"type": "impute_mean", "scope": {"columns": ["age", "income"]}},
            {"type": "clip_iqr",    "scope": {"columns": ["income"]}},
            {"type": "standard_scaler", "scope": {"columns": ["age", "income"]}}
        ]
    }
    """
    async def _exec():
        cfg = load_config(pipeline_file)
        raw_steps = cfg.get("steps", [])
        if not raw_steps:
            console.print("[red]Pipeline config has no steps[/red]")
            raise typer.Exit(1)

        steps = [
            StepConfig(step=s["type"], columns=s.get("scope", {}).get("columns") or [])
            for s in raw_steps
        ]

        pipeline = PreprocessPipeline(steps=steps, checkpoint=checkpoint)
        ctx = make_context()
        validate_pipeline(pipeline, ctx)
        plan = pipeline.plan(ctx)
        plan.metadata["dry_run"] = dry_run
        plan.metadata["input_artifact_id"] = resolve_input(ctx, input, feature_hint="preprocess")

        with Progress(
            SpinnerColumn(), BarColumn(),
            TextColumn("{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as p:
            task = p.add_task("Preprocessing...", total=100)
            async for event in pipeline.execute(plan, ctx):
                p.update(task, completed=int((event.progress_pct or 0) * 100), description=event.message)
                if event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    console.print(f"\n[green][OK] {event.message}[/green]")
                    if dry_run:
                        console.print("  [yellow]Dry run - no artifact saved[/yellow]")
                    else:
                        console.print(f"  Output:  [cyan]{payload.get('output_artifact_id')}[/cyan]")
                        console.print(f"  Config:  [cyan]{payload.get('config_artifact_id')}[/cyan]")
                    console.print(f"  Run ID:  [cyan]{payload.get('run_id')}[/cyan]")
                elif event.event_type == EventType.ERROR:
                    console.print(f"\n[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

        if pipeline.run_id and hasattr(pipeline, "_detect_conflicts"):
            for w in pipeline._detect_conflicts():
                console.print(f"[yellow]Warning: {w}[/yellow]")

    run_async(_exec())