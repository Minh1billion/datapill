import json
from typing import Optional

import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from datapill.core.events import EventType
from datapill.features.classify.pipeline import ClassifyPipeline
from datapill.features.classify.schema import ClassifyConfig

from ._shared import console, make_context, resolve_input, run_async, validate_pipeline
from ._tables import print_classify_table

app = typer.Typer()


@app.command("classify")
def cmd_classify(
    input: str = typer.Option(..., "--input", "-i", help="run_id or full artifact ID"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="rule_based | embedding | hybrid"),
    threshold: float = typer.Option(0.0, "--threshold", "-t", help="Minimum confidence (0.0–1.0)"),
    overrides: Optional[str] = typer.Option(None, "--overrides", help='JSON string: {"col_name": "semantic_type"}'),
):
    """Classify columns in a dataset by semantic type.

    Modes:
      rule_based  - fast, regex + dtype heuristics only
      embedding   - semantic similarity via sentence-transformers
      hybrid      - rule_based first, embedding for ambiguous columns (default)

    Examples:

      dp classify -i <run_id> --mode hybrid

      dp classify -i <run_id> --mode rule_based --threshold 0.65

      dp classify -i <run_id> --overrides '{"age": "numerical_continuous", "y": "target_label"}'
    """
    async def _exec():
        override_dict: dict = {}
        if overrides:
            try:
                override_dict = json.loads(overrides)
            except json.JSONDecodeError:
                console.print("[red]--overrides must be valid JSON, e.g. '{\"col\": \"boolean\"}'[/red]")
                raise typer.Exit(1)

        pipeline = ClassifyPipeline(ClassifyConfig(
            mode=mode,
            confidence_threshold=threshold,
            overrides=override_dict,
        ))
        ctx = make_context()
        validate_pipeline(pipeline, ctx)
        plan = pipeline.plan(ctx)
        plan.metadata["input_artifact_id"] = resolve_input(ctx, input, feature_hint="classify")

        with Progress(
            SpinnerColumn(), BarColumn(),
            TextColumn("{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as p:
            task = p.add_task("Classifying...", total=100)
            async for event in pipeline.execute(plan, ctx):
                p.update(task, completed=int((event.progress_pct or 0) * 100), description=event.message)
                if event.event_type == EventType.ERROR:
                    console.print(f"\n[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)
                if event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    console.print(f"\n[green][OK] {event.message}[/green]")
                    console.print(f"  Artifact: [cyan]{payload.get('output_artifact_id')}[/cyan]")
                    console.print(f"  Columns:  [cyan]{payload.get('column_count')}[/cyan]")
                    await print_classify_table(ctx, payload.get("output_artifact_id"))

    run_async(_exec())