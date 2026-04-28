from pathlib import Path
from typing import Optional

import typer

from datapill.features.export.codegen import CodegenConfig, generate

from ._shared import FORMATS, SOURCES, console, load_config, make_context, run_async

app = typer.Typer(help="Pipeline utilities")


@app.command("export")
def cmd_pipeline_export(
    input: str = typer.Option(..., "--input", "-i", help="run_id or artifact ID of a preprocess run"),
    ingest_config: Optional[str] = typer.Option(None, "--ingest-config", "-c", help="Connector JSON config (same as dp ingest --config)"),
    ingest_source: str = typer.Option("local_file", "--source", "-s", help=f"Connector type: {SOURCES}"),
    ingest_path: Optional[str] = typer.Option(None, "--path", help="File path (local_file)"),
    ingest_table: Optional[str] = typer.Option(None, "--table", help="Table name (postgresql | mysql)"),
    ingest_url: Optional[str] = typer.Option(None, "--url", help="S3 URL"),
    output_format: str = typer.Option("parquet", "--format", "-f", help=f"Output format: {FORMATS}"),
    output_path: str = typer.Option("output/result.parquet", "--out-path", help="Output path hard-coded into the script"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Name for generated files, e.g. orders_pipeline → run_orders_pipeline.py"),
    compression: Optional[str] = typer.Option(None, "--compression", help="snappy | zstd | gzip (parquet only)"),
    with_tests: bool = typer.Option(False, "--with-tests", help="Also generate test_pipeline.py"),
    out_dir: str = typer.Option("generated", "--out-dir", "-o", help="Directory to write generated files"),
    artifact_store: str = typer.Option("src/datapill/artifacts", "--store", help="Artifact store directory"),
):
    """Generate a standalone run_pipeline.py from a preprocess run.

    Reads the preprocess config artifact to recover the step list, merges it
    with the ingest config you provide, then writes a self-contained script
    that has no dependency on datapill.

    \b
    Examples:

      dp pipeline export -i def456 -s postgresql -c pg.json

      dp pipeline export -i def456 -s local_file --path data.csv --with-tests

      dp pipeline export -i def456_preprocess_config --out-dir ./generated
    """
    async def _exec():
        ctx = make_context(artifact_store)

        try:
            resolved = ctx.artifact_store.resolve(input, feature_hint="preprocess")
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

        if resolved.endswith("_preprocess_output"):
            config_id = resolved.replace("_preprocess_output", "_preprocess_config")
            if ctx.artifact_store.exists(config_id, "json"):
                resolved = config_id
            else:
                console.print("[red]Cannot find preprocess config. Run dp preprocess without --dry-run first.[/red]")
                raise typer.Exit(1)

        try:
            preprocess_json = await ctx.artifact_store.load_json(resolved)
        except FileNotFoundError:
            console.print(f"[red]Artifact not found: {resolved}[/red]")
            raise typer.Exit(1)

        raw_steps = preprocess_json.get("steps", [])
        if not raw_steps:
            console.print("[red]Preprocess config has no steps. Nothing to generate.[/red]")
            raise typer.Exit(1)

        codegen_steps = [
            {
                "step": s["type"],
                "columns": s.get("scope", {}).get("columns") or [],
                "params": s.get("params", {}),
            }
            for s in raw_steps
        ]

        ingest_cfg: dict = load_config(ingest_config)
        ingest_cfg["source"] = ingest_source
        if ingest_path:
            ingest_cfg["path"] = ingest_path
        if ingest_table:
            ingest_cfg["table"] = ingest_table
        if ingest_url:
            ingest_cfg["url"] = ingest_url

        if ingest_source == "local_file" and not ingest_cfg.get("path"):
            ingest_cfg["path"] = "<FILL_IN: path to your source file>"
            console.print("[yellow]Warning: --path not provided. Fill in INGEST_CONFIG['path'] in the generated script.[/yellow]")

        output_cfg: dict = {"format": output_format, "path": output_path}
        if compression:
            output_cfg["compression"] = compression

        label = name or resolved.split("_")[0]

        generated_files = generate(
            CodegenConfig(ingest=ingest_cfg, preprocess=codegen_steps, output=output_cfg, with_tests=with_tests),
            Path(out_dir),
            name=label,
        )

        console.print(f"\n[green][OK] Generated {len(generated_files)} file(s) → {out_dir}/[/green]")
        console.print(f"  Source artifact : [cyan]{resolved}[/cyan]")
        console.print(f"  Steps           : [cyan]{len(codegen_steps)}[/cyan]")
        for f in generated_files:
            console.print(f"  [bold]{f.name}[/bold]  ({f.stat().st_size:,} bytes)")
        console.print("\n[dim]Run:[/dim]")
        for f in generated_files:
            if f.name.startswith("run_"):
                console.print(f"  [bold]python {out_dir}/{f.name} --dry-run[/bold]")
                console.print(f"  [bold]python {out_dir}/{f.name}[/bold]")
            elif f.name.startswith("test_"):
                console.print(f"  [bold]python -m pytest {out_dir}/{f.name} -v[/bold]")

    run_async(_exec())