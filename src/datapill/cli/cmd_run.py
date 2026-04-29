import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from datapill.core.events import EventType
from datapill.features.ingest.pipeline import IngestConfig, IngestPipeline
from datapill.features.profile.pipeline import ProfileOptions, ProfilePipeline

from ._shared import build_connector, console, load_config, make_context, run_async, validate_pipeline
from ._tables import print_profile_table

app = typer.Typer()


@app.command("run")
def cmd_run(
    config_file: str = typer.Argument(..., help="Path to pipeline JSON config file"),
):
    """Run a full ingest + profile pipeline from a JSON config file.

    Config format:
    {
        "source": "postgresql",
        "connector": { "host": "...", "database": "...", "user": "...", "password": "..." },
        "query":   { "table": "orders" },
        "ingest":  { "batch_size": 10000 },
        "profile": { "mode": "full", "correlation": "pearson" }
    }
    """
    async def _exec():
        cfg = load_config(config_file)
        source = cfg.get("source")
        if not source:
            console.print("[red]'source' is required in config[/red]")
            raise typer.Exit(1)

        raw_query: dict = cfg.get("query", {})
        connector, query = build_connector(
            source, cfg.get("connector", {}),
            path=raw_query.get("path"),
            table=raw_query.get("table"),
            url=raw_query.get("url"),
            topic=raw_query.get("topic"),
            endpoint=raw_query.get("endpoint"),
        )

        ctx = make_context()

        console.rule("[bold]Ingest")
        ingest_pipeline = IngestPipeline(IngestConfig(
            connector=connector,
            query=query,
            options=cfg.get("ingest", {}),
        ))
        validate_pipeline(ingest_pipeline, ctx)
        ingest_plan = ingest_pipeline.plan(ctx)
        output_artifact_id: str | None = None

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:
            task = p.add_task("Ingesting...", total=None)
            async for event in ingest_pipeline.execute(ingest_plan, ctx):
                p.update(task, description=event.message)
                if event.event_type == EventType.DONE:
                    output_artifact_id = (event.payload or {}).get("output_artifact_id")
                    console.print(f"[green][OK] {event.message}[/green]")
                elif event.event_type == EventType.ERROR:
                    console.print(f"[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

        await connector.close()

        if not output_artifact_id:
            console.print("[yellow]No artifact produced, skipping profile.[/yellow]")
            return

        profile_opts: dict = cfg.get("profile", {})
        console.rule("[bold]Profile")
        profile_pipeline = ProfilePipeline(ProfileOptions(
            mode=profile_opts.get("mode", "full"),
            sample_strategy=profile_opts.get("sample_strategy", "none"),
            sample_size=profile_opts.get("sample_size", 100_000),
            correlation_method=profile_opts.get("correlation", "pearson"),
        ))
        validate_pipeline(profile_pipeline, ctx)
        profile_plan = profile_pipeline.plan(ctx)
        profile_plan.metadata["input_artifact_id"] = output_artifact_id

        with Progress(
            SpinnerColumn(), BarColumn(),
            TextColumn("{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as p:
            task = p.add_task("Profiling...", total=100)
            async for event in profile_pipeline.execute(profile_plan, ctx):
                p.update(task, completed=int((event.progress_pct or 0) * 100), description=event.message)
                if event.event_type == EventType.DONE:
                    console.print(f"[green][OK] {event.message}[/green]")
                    await print_profile_table(ctx, (event.payload or {}).get("summary_artifact_id"))
                elif event.event_type == EventType.ERROR:
                    console.print(f"[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

    run_async(_exec())