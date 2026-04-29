import json
import os
from pathlib import Path
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
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="run_id or full artifact ID of a profile result (detail or summary)"),
    model_cache_dir: Optional[str] = typer.Option(None, "--model-cache-dir", help="Directory to cache embedding model (default: ~/.cache/datapill/models). Set DATAPILL_MODEL_CACHE env var as alternative."),
):
    """Classify columns in a dataset by semantic type.

    Modes:
      rule_based  - fast, regex + dtype heuristics only
      embedding   - semantic similarity via fastembed (BAAI/bge-small-en-v1.5)
      hybrid      - rule_based first, embedding for ambiguous columns (default)

    NOTE: modes 'embedding' and 'hybrid' download ~130 MB on first use.
    Use --model-cache-dir to control where the model is cached, or set
    DATAPILL_MODEL_CACHE. Use --mode rule_based for fully offline operation.

    Providing --profile improves classification accuracy by using pre-computed
    statistics (null rates, cardinality, detected patterns, skewness) as additional
    signals for all modes.

    Examples:

      dp classify -i <run_id> --mode hybrid

      dp classify -i <run_id> --profile <profile_run_id>

      dp classify -i <run_id> --mode rule_based --threshold 0.65

      dp classify -i <run_id> --overrides '{"age": "numerical_continuous", "y": "target_label"}'

      dp classify -i <run_id> --mode embedding --model-cache-dir /mnt/models
    """
    async def _exec():
        override_dict: dict = {}
        if overrides:
            try:
                override_dict = json.loads(overrides)
            except json.JSONDecodeError:
                console.print("[red]--overrides must be valid JSON, e.g. '{\"col\": \"boolean\"}'[/red]")
                raise typer.Exit(1)

        if not profile:
            console.print(
                "[yellow]Tip: run [bold]dp profile -i <run_id>[/bold] first and pass [bold]--profile <run_id>[/bold] "
                "to improve classification quality with pre-computed statistics.[/yellow]"
            )

        # Resolve model cache directory: CLI flag > env var > default.
        resolved_cache = (
            model_cache_dir
            or os.environ.get("DATAPILL_MODEL_CACHE")
            or str(Path.home() / ".cache" / "datapill" / "models")
        )

        if mode in ("embedding", "hybrid"):
            cache_path = Path(resolved_cache)
            model_present = cache_path.is_dir() and any(cache_path.iterdir())
            if not model_present:
                console.print(
                    f"[yellow]⚠ Mode '{mode}' requires the BAAI/bge-small-en-v1.5 embedding model "
                    f"(~130 MB). It will be downloaded on first use to:[/yellow]\n"
                    f"  [cyan]{resolved_cache}[/cyan]\n"
                    "[yellow]Use [bold]--mode rule_based[/bold] for fully offline operation.[/yellow]"
                )

        pipeline = ClassifyPipeline(ClassifyConfig(
            mode=mode,
            confidence_threshold=threshold,
            overrides=override_dict,
            model_cache_dir=resolved_cache,
        ))
        ctx = make_context()
        validate_pipeline(pipeline, ctx)
        plan = pipeline.plan(ctx)
        plan.metadata["input_artifact_id"] = resolve_input(ctx, input, feature_hint="classify")

        if profile:
            plan.metadata["profile_artifact_id"] = profile

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
                    console.print(f"  Artifact:     [cyan]{payload.get('output_artifact_id')}[/cyan]")
                    console.print(f"  Columns:      [cyan]{payload.get('column_count')}[/cyan]")
                    console.print(f"  Profile used: [cyan]{payload.get('profile_used')}[/cyan]")
                    await print_classify_table(ctx, payload.get("output_artifact_id"))

    run_async(_exec())