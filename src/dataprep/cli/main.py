import asyncio
import json
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from dataprep.connectors.base import BaseConnector
from dataprep.connectors.kafka import KafkaConnector
from dataprep.connectors.local_file import LocalFileConnector
from dataprep.connectors.mysql import MySQLConnector
from dataprep.connectors.postgresql import PostgreSQLConnector
from dataprep.connectors.rest import RESTConnector
from dataprep.connectors.s3 import S3Connector
from dataprep.core.context import PipelineContext
from dataprep.core.events import EventType
from dataprep.features.export.pipeline import ExportPipeline, WriteConfig
from dataprep.features.ingest.pipeline import IngestConfig, IngestPipeline
from dataprep.features.preprocess.pipeline import PreprocessPipeline
from dataprep.features.preprocess.schema import StepConfig
from dataprep.features.profile.pipeline import ProfileOptions, ProfilePipeline
from dataprep.storage.artifact import ArtifactStore

app = typer.Typer(name="dp", help="DataPrep CLI", no_args_is_help=True)
console = Console()

_SOURCES = "local_file | postgresql | mysql | s3 | rest | kafka"
_FORMATS = "csv | parquet | json | jsonl | excel"


def _run(coro):
    return asyncio.run(coro)


def _make_context(out: str) -> PipelineContext:
    return PipelineContext(artifact_store=ArtifactStore(out))


def _load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise typer.Exit(1)
    with p.open() as f:
        return json.load(f)


def _build_connector(
    source: str,
    config: dict,
    path: str | None = None,
    table: str | None = None,
    url: str | None = None,
    topic: str | None = None,
    endpoint: str | None = None,
) -> tuple[BaseConnector, dict]:
    if source == "local_file":
        if not path:
            console.print("[red]--path is required for local_file[/red]")
            raise typer.Exit(1)
        return LocalFileConnector(config), {"path": path}

    if source == "postgresql":
        _require_db_config(config, source)
        query: dict = {}
        if table:
            query["table"] = table
        elif "sql" in config:
            query["sql"] = config.pop("sql")
        else:
            console.print("[red]--table or 'sql' in config is required for postgresql[/red]")
            raise typer.Exit(1)
        return PostgreSQLConnector(config), query

    if source == "mysql":
        _require_db_config(config, source)
        query = {}
        if table:
            query["table"] = table
        elif "sql" in config:
            query["sql"] = config.pop("sql")
        else:
            console.print("[red]--table or 'sql' in config is required for mysql[/red]")
            raise typer.Exit(1)
        return MySQLConnector(config), query

    if source == "s3":
        if not url and not config.get("default_url"):
            console.print("[red]--url or 'default_url' in config is required for s3[/red]")
            raise typer.Exit(1)
        return S3Connector(config), {"url": url or config["default_url"]}

    if source == "rest":
        if not endpoint and not config.get("default_endpoint"):
            console.print("[red]--endpoint or 'default_endpoint' in config is required for rest[/red]")
            raise typer.Exit(1)
        if "base_url" not in config:
            console.print("[red]'base_url' is required in config for rest[/red]")
            raise typer.Exit(1)
        ep = endpoint or config["default_endpoint"]
        return RESTConnector(config), {"endpoint": ep}

    if source == "kafka":
        if not topic and not config.get("default_topic"):
            console.print("[red]--topic or 'default_topic' in config is required for kafka[/red]")
            raise typer.Exit(1)
        if "bootstrap_servers" not in config:
            console.print("[red]'bootstrap_servers' is required in config for kafka[/red]")
            raise typer.Exit(1)
        t = topic or config["default_topic"]
        return KafkaConnector(config), {"topic": t}

    console.print(f"[red]Unknown source: {source}. Available: {_SOURCES}[/red]")
    raise typer.Exit(1)


def _require_db_config(config: dict, source: str) -> None:
    required = ["host", "database", "user", "password"]
    missing = [k for k in required if k not in config]
    if missing:
        console.print(f"[red]{source} config missing keys: {missing}[/red]")
        raise typer.Exit(1)


@app.command("ingest")
def cmd_ingest(
    source: str = typer.Option(..., "--source", "-s", help=f"Connector type: {_SOURCES}"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to JSON config file for connector"),
    path: Optional[str] = typer.Option(None, "--path", help="File path (local_file)"),
    table: Optional[str] = typer.Option(None, "--table", help="Table name (postgresql | mysql)"),
    url: Optional[str] = typer.Option(None, "--url", help="S3 URL, e.g. s3://bucket/key.parquet"),
    topic: Optional[str] = typer.Option(None, "--topic", help="Kafka topic name"),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", help="REST endpoint, e.g. /users"),
    out: str = typer.Option("src/dataprep/artifacts", "--out", "-o", help="Artifact output directory"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max rows to read"),
    batch_size: int = typer.Option(50_000, "--batch-size", help="Rows per batch"),
    max_records: Optional[int] = typer.Option(None, "--max-records", help="Max records (kafka)"),
):
    """Ingest data from a connector into artifact store."""
    async def _run_ingest():
        config = _load_config(config_file)
        connector, query = _build_connector(source, config, path, table, url, topic, endpoint)

        options: dict = {"batch_size": batch_size}
        if limit:
            options["n_rows"] = limit
        if max_records:
            options["max_records"] = max_records

        pipeline = IngestPipeline(IngestConfig(connector=connector, query=query, options=options))
        ctx = _make_context(out)

        result = pipeline.validate(ctx)
        if not result.ok:
            for err in result.errors:
                console.print(f"[red]Validation error: {err}[/red]")
            raise typer.Exit(1)

        plan = pipeline.plan(ctx)

        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console
        ) as progress:
            task = progress.add_task("Ingesting...", total=None)
            async for event in pipeline.execute(plan, ctx):
                progress.update(task, description=event.message)
                if event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    console.print(f"\n[green][OK] {event.message}[/green]")
                    console.print(f"  Artifact: [cyan]{payload.get('output_artifact_id')}[/cyan]")
                    console.print(f"  Schema:   [cyan]{payload.get('schema_artifact_id')}[/cyan]")
                    console.print(f"  Rows:     [cyan]{payload.get('rows_read', 0):,}[/cyan]")
                elif event.event_type == EventType.ERROR:
                    console.print(f"\n[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

    _run(_run_ingest())


@app.command("profile")
def cmd_profile(
    input: str = typer.Option(..., "--input", "-i", help="Artifact ID or .parquet file path"),
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
                    console.print(f"\n[green][OK] {event.message}[/green]")
                    console.print(f"  Profile ID: [cyan]{payload.get('profile_id')}[/cyan]")
                    console.print(f"  Detail:     [cyan]{payload.get('detail_artifact_id')}[/cyan]")
                    console.print(f"  Summary:    [cyan]{payload.get('summary_artifact_id')}[/cyan]")
                    await _print_profile_table(ctx, payload.get("summary_artifact_id"))
                elif event.event_type == EventType.ERROR:
                    console.print(f"\n[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

    _run(_run_profile())


@app.command("preprocess")
def cmd_preprocess(
    input: str = typer.Option(..., "--input", "-i", help="Artifact ID or .parquet file path"),
    pipeline_file: str = typer.Option(..., "--pipeline", "-p", help="Path to pipeline JSON config file"),
    out: str = typer.Option("src/dataprep/artifacts", "--out", "-o", help="Artifact output directory"),
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
    async def _run_preprocess():
        cfg = _load_config(pipeline_file)
        raw_steps = cfg.get("steps", [])
        if not raw_steps:
            console.print("[red]Pipeline config has no steps[/red]")
            raise typer.Exit(1)

        steps = [
            StepConfig(
                step=s["type"],
                columns=s.get("scope", {}).get("columns") or [],
            )
            for s in raw_steps
        ]

        pipeline = PreprocessPipeline(steps=steps, checkpoint=checkpoint)
        ctx = _make_context(out)

        result = pipeline.validate(ctx)
        if not result.ok:
            for err in result.errors:
                console.print(f"[red]Validation error: {err}[/red]")
            raise typer.Exit(1)

        plan = pipeline.plan(ctx)
        plan.metadata["dry_run"] = dry_run

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
            task = progress.add_task("Preprocessing...", total=100)
            async for event in pipeline.execute(plan, ctx):
                pct = int((event.progress_pct or 0) * 100)
                progress.update(task, completed=pct, description=event.message)
                if event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    console.print(f"\n[green][OK] {event.message}[/green]")
                    if dry_run:
                        console.print("  [yellow]Dry run — no artifact saved[/yellow]")
                    else:
                        console.print(f"  Output:  [cyan]{payload.get('output_artifact_id')}[/cyan]")
                        console.print(f"  Config:  [cyan]{payload.get('config_artifact_id')}[/cyan]")
                    console.print(f"  Run ID:  [cyan]{payload.get('run_id')}[/cyan]")
                elif event.event_type == EventType.ERROR:
                    console.print(f"\n[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

        if pipeline.run_id and hasattr(pipeline, '_detect_conflicts'):
            warnings = pipeline._detect_conflicts()
            for w in warnings:
                console.print(f"[yellow]Warning: {w}[/yellow]")

    _run(_run_preprocess())


@app.command("export")
def cmd_export(
    input: str = typer.Option(..., "--input", "-i", help="Artifact ID or .parquet file path"),
    format: str = typer.Option(..., "--format", "-f", help=f"Output format: {_FORMATS}"),
    out_path: Optional[str] = typer.Option(None, "--out-path", help="Output file path"),
    write_mode: str = typer.Option("replace", "--write-mode", help="replace | append | upsert"),
    primary_keys: Optional[str] = typer.Option(None, "--primary-keys", help="Comma-separated keys for upsert"),
    connector_file: Optional[str] = typer.Option(None, "--connector", "-c", help="Connector config JSON for write-back"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print first 10 rows, skip write"),
    compression: Optional[str] = typer.Option(None, "--compression", help="Compression (parquet: snappy|zstd|gzip)"),
    out: str = typer.Option("src/dataprep/artifacts", "--out", "-o", help="Artifact store directory"),
):
    """Export a dataset to file or write back to a connector.

    Export to file:
        dp export -i <artifact_id> -f parquet --out-path output/result.parquet

    Write back to PostgreSQL:
        dp export -i <artifact_id> -f parquet --connector pg.json --write-mode upsert --primary-keys id
    """
    async def _run_export():
        connector_config: dict | None = None
        if connector_file:
            connector_config = _load_config(connector_file)
            if not connector_config.get("source"):
                console.print("[red]connector config must have 'source' field (postgresql | mysql | s3)[/red]")
                raise typer.Exit(1)

        if connector_config is None and not out_path:
            console.print("[red]--out-path is required when not using --connector[/red]")
            raise typer.Exit(1)

        options: dict = {}
        if compression:
            options["compression"] = compression

        pkeys = [k.strip() for k in primary_keys.split(",")] if primary_keys else []

        cfg = WriteConfig(
            format=format,
            path=Path(out_path) if out_path else None,
            options=options,
            write_mode=write_mode,
            primary_keys=pkeys,
            connector_config=connector_config,
        )

        pipeline = ExportPipeline(cfg)
        ctx = _make_context(out)

        result = pipeline.validate(ctx)
        if not result.ok:
            for err in result.errors:
                console.print(f"[red]Validation error: {err}[/red]")
            raise typer.Exit(1)

        plan = pipeline.plan(ctx)
        plan.metadata["dry_run"] = dry_run

        if Path(input).exists() and Path(input).suffix == ".parquet":
            plan.metadata["dataframe"] = pl.read_parquet(input)
        else:
            plan.metadata["input_artifact_id"] = input

        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console
        ) as progress:
            task = progress.add_task("Exporting...", total=None)
            async for event in pipeline.execute(plan, ctx):
                progress.update(task, description=event.message)
                if event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    console.print(f"\n[green][OK] {event.message}[/green]")
                    if dry_run:
                        console.print("  [yellow]Dry run — no data written[/yellow]")
                    else:
                        console.print(f"  Destination: [cyan]{payload.get('destination')}[/cyan]")
                        console.print(f"  Rows:        [cyan]{payload.get('rows_written', 0):,}[/cyan]")
                elif event.event_type == EventType.ERROR:
                    console.print(f"\n[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

    _run(_run_export())


@app.command("connector")
def cmd_connector(
    action: str = typer.Argument(..., help="test | schema | upload | download | list | exec | truncate | produce"),
    source: str = typer.Option(..., "--source", "-s", help=f"Connector type: {_SOURCES}"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to JSON config file"),
    # shared
    path: Optional[str] = typer.Option(None, "--path", help="Local file path"),
    table: Optional[str] = typer.Option(None, "--table", help="Table name (postgresql | mysql)"),
    url: Optional[str] = typer.Option(None, "--url", help="S3 URL (s3://bucket/key)"),
    topic: Optional[str] = typer.Option(None, "--topic", help="Kafka topic name"),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", help="REST endpoint"),
    # upload / download
    src_path: Optional[str] = typer.Option(None, "--src-path", help="Source local file path (upload)"),
    dest_url: Optional[str] = typer.Option(None, "--dest-url", help="Destination S3 URL (upload)"),
    out_path: Optional[str] = typer.Option(None, "--out-path", help="Local output path (download)"),
    # list
    prefix: Optional[str] = typer.Option(None, "--prefix", help="S3 key prefix filter (list)"),
    # exec
    sql: Optional[str] = typer.Option(None, "--sql", help="Raw SQL to execute (exec)"),
    # produce
    file: Optional[str] = typer.Option(None, "--file", help="JSON/CSV file to produce to Kafka"),
    key_column: Optional[str] = typer.Option(None, "--key-column", help="Column to use as Kafka message key"),
):
    """Interact with a connector: test | schema | upload | download | list | exec | truncate | produce.

    Examples:

      dp connector upload   --source s3 --config s3.json --src-path data.csv --dest-url s3://bucket/data.csv

      dp connector download --source s3 --config s3.json --url s3://bucket/data.csv --out-path ./data.csv

      dp connector list     --source s3 --config s3.json --prefix input/

      dp connector exec     --source postgresql --config pg.json --sql "DELETE FROM orders WHERE status='cancelled'"

      dp connector truncate --source postgresql --config pg.json --table orders

      dp connector produce  --source kafka --config kafka.json --topic my-topic --file records.json
    """
    async def _run_connector():
        config = _load_config(config_file)

        if source == "local_file":
            connector: BaseConnector = LocalFileConnector({"default_path": path or "."})
        elif source == "postgresql":
            _require_db_config(config, source)
            if table:
                config["default_table"] = table
            connector = PostgreSQLConnector(config)
        elif source == "mysql":
            _require_db_config(config, source)
            if table:
                config["default_table"] = table
            connector = MySQLConnector(config)
        elif source == "s3":
            if url:
                config["default_url"] = url
            connector = S3Connector(config)
        elif source == "rest":
            if endpoint:
                config["default_endpoint"] = endpoint
            connector = RESTConnector(config)
        elif source == "kafka":
            if topic:
                config["default_topic"] = topic
            connector = KafkaConnector(config)
        else:
            console.print(f"[red]Unknown source: {source}. Available: {_SOURCES}[/red]")
            raise typer.Exit(1)
        
        if action == "test":
            status = await connector.test_connection()
            if status.ok:
                ms = f"{status.latency_ms:.1f}ms" if status.latency_ms is not None else "n/a"
                console.print(f"[green][OK] Connection OK ({ms})[/green]")
            else:
                console.print(f"[red][ERROR] Connection failed: {status.error}[/red]")

        elif action == "schema":
            schema = await connector.schema()
            if not schema.columns:
                console.print("[yellow]No schema returned (topic/table/path may be empty or not configured)[/yellow]")
                return
            title = path or table or url or topic or endpoint or source
            t = Table(title=f"Schema: {title}", show_lines=True)
            t.add_column("Column", style="bold")
            t.add_column("Type")
            t.add_column("Nullable")
            for col in schema.columns:
                t.add_row(col.name, col.dtype, "yes" if col.nullable else "no")
            if schema.row_count_estimate is not None:
                console.print(f"  Row estimate: [cyan]{schema.row_count_estimate:,}[/cyan]")
            console.print(t)

        elif action == "upload":
            if source not in ("s3", "local_file"):
                console.print("[red]upload is only supported for s3 and local_file[/red]")
                raise typer.Exit(1)
            if not src_path:
                console.print("[red]--src-path is required for upload[/red]")
                raise typer.Exit(1)

            src = Path(src_path)
            if not src.exists():
                console.print(f"[red]Source file not found: {src_path}[/red]")
                raise typer.Exit(1)

            fmt = src.suffix.lstrip(".").lower() or "csv"
            df = _read_local(src, fmt)

            if source == "s3":
                if not dest_url:
                    console.print("[red]--dest-url is required for s3 upload (e.g. s3://bucket/key.csv)[/red]")
                    raise typer.Exit(1)
                result = await connector.write(df, {"url": dest_url})
            else:
                if not dest_url:
                    console.print("[red]--dest-url is required (local destination path)[/red]")
                    raise typer.Exit(1)
                result = await connector.write(df, {"path": dest_url})

            console.print(f"[green][OK] Uploaded {result.rows_written:,} rows in {result.duration_s:.2f}s[/green]")
            console.print(f"  Destination: [cyan]{dest_url}[/cyan]")

        elif action == "download":
            if source != "s3":
                console.print("[red]download is only supported for s3[/red]")
                raise typer.Exit(1)
            if not url:
                console.print("[red]--url is required for download (e.g. s3://bucket/key.parquet)[/red]")
                raise typer.Exit(1)
            if not out_path:
                console.print("[red]--out-path is required for download[/red]")
                raise typer.Exit(1)

            with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:
                task = p.add_task("Downloading...", total=None)
                df = await connector.read({"url": url})
                p.update(task, description=f"Downloaded {len(df):,} rows")

            dest = Path(out_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            fmt = dest.suffix.lstrip(".").lower() or "parquet"
            _write_local(df, dest, fmt)

            console.print(f"[green][OK] Downloaded {len(df):,} rows → {out_path}[/green]")

        elif action == "list":
            if source != "s3":
                console.print("[red]list is only supported for s3[/red]")
                raise typer.Exit(1)

            bucket = config.get("bucket")
            if not bucket:
                console.print("[red]'bucket' is required in s3 config for list[/red]")
                raise typer.Exit(1)

            import aioboto3
            from urllib.parse import urlparse

            session = aioboto3.Session(
                aws_access_key_id=config.get("aws_access_key_id"),
                aws_secret_access_key=config.get("aws_secret_access_key"),
                region_name=config.get("region", "us-east-1"),
            )
            client_kwargs = {}
            if endpoint_url := config.get("endpoint_url"):
                client_kwargs["endpoint_url"] = endpoint_url

            t = Table(title=f"s3://{bucket}/{prefix or ''}", show_lines=True)
            t.add_column("Key", style="bold")
            t.add_column("Size", justify="right")
            t.add_column("Last Modified")

            total = 0
            async with session.client("s3", **client_kwargs) as client:
                paginator = client.get_paginator("list_objects_v2")
                async for page in paginator.paginate(Bucket=bucket, Prefix=prefix or ""):
                    for obj in page.get("Contents", []):
                        size = obj["Size"]
                        size_str = f"{size / 1024:.1f} KB" if size < 1_048_576 else f"{size / 1_048_576:.1f} MB"
                        t.add_row(obj["Key"], size_str, str(obj["LastModified"])[:19])
                        total += 1

            console.print(t)
            console.print(f"  Total: [cyan]{total}[/cyan] objects")

        elif action == "exec":
            if source not in ("postgresql", "mysql"):
                console.print("[red]exec is only supported for postgresql and mysql[/red]")
                raise typer.Exit(1)
            if not sql:
                console.print("[red]--sql is required for exec[/red]")
                raise typer.Exit(1)

            await _db_exec(source, config, sql)

        elif action == "truncate":
            if source not in ("postgresql", "mysql"):
                console.print("[red]truncate is only supported for postgresql and mysql[/red]")
                raise typer.Exit(1)
            if not table:
                console.print("[red]--table is required for truncate[/red]")
                raise typer.Exit(1)

            if source == "postgresql":
                await _db_exec(source, config, f'TRUNCATE TABLE "{table}"')
            else:
                await _db_exec(source, config, f"TRUNCATE TABLE `{table}`")

        elif action == "produce":
            if source != "kafka":
                console.print("[red]produce is only supported for kafka[/red]")
                raise typer.Exit(1)
            if not topic:
                console.print("[red]--topic is required for produce[/red]")
                raise typer.Exit(1)
            if not file:
                console.print("[red]--file is required for produce (JSON or CSV file)[/red]")
                raise typer.Exit(1)

            src = Path(file)
            if not src.exists():
                console.print(f"[red]File not found: {file}[/red]")
                raise typer.Exit(1)

            fmt = src.suffix.lstrip(".").lower()
            df = _read_local(src, fmt)

            opts: dict = {}
            if key_column:
                opts["key_column"] = key_column

            with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:
                task = p.add_task(f"Producing to {topic}...", total=None)
                result = await connector.write(df, {"topic": topic}, opts)
                p.update(task, description=f"Produced {result.rows_written:,} messages")

            console.print(f"[green][OK] Produced {result.rows_written:,} messages to '{topic}' in {result.duration_s:.2f}s[/green]")

        else:
            console.print(f"[red]Unknown action: {action}. Use: test | schema | upload | download | list | exec | truncate | produce[/red]")
            raise typer.Exit(1)

        await connector.close()

    _run(_run_connector())


def _read_local(path: Path, fmt: str) -> pl.DataFrame:
    fmt = fmt.lower()
    if fmt == "csv":
        return pl.read_csv(path)
    if fmt == "parquet":
        return pl.read_parquet(path)
    if fmt in ("json",):
        return pl.read_json(path)
    if fmt in ("jsonl", "ndjson"):
        return pl.read_ndjson(path)
    if fmt in ("xlsx", "excel"):
        return pl.read_excel(path)
    raise ValueError(f"Unsupported format for read: '{fmt}'")


def _write_local(df: pl.DataFrame, path: Path, fmt: str) -> None:
    fmt = fmt.lower()
    if fmt == "csv":
        df.write_csv(path)
    elif fmt == "parquet":
        df.write_parquet(path)
    elif fmt in ("json",):
        df.write_json(path)
    elif fmt in ("jsonl", "ndjson"):
        df.write_ndjson(path)
    elif fmt in ("xlsx", "excel"):
        df.write_excel(path)
    else:
        raise ValueError(f"Unsupported format for write: '{fmt}'")


async def _db_exec(source: str, config: dict, sql: str) -> None:
    if source == "postgresql":
        import asyncpg
        conn = await asyncpg.connect(
            host=config["host"], port=config.get("port", 5432),
            database=config["database"], user=config["user"], password=config["password"],
        )
        try:
            result = await conn.execute(sql)
            console.print(f"[green][OK] {result}[/green]")
        finally:
            await conn.close()

    elif source == "mysql":
        import asyncmy
        import asyncmy.cursors
        conn = await asyncmy.connect(
            host=config["host"], port=config.get("port", 3306),
            db=config["database"], user=config["user"], password=config["password"],
        )
        try:
            async with conn.cursor(asyncmy.cursors.DictCursor) as cur:
                await cur.execute(sql)
                affected = cur.rowcount
            await conn.commit()
            console.print(f"[green][OK] {affected} row(s) affected[/green]")
        finally:
            conn.close()


@app.command("run")
def cmd_run(
    config_file: str = typer.Argument(..., help="Path to pipeline JSON config file"),
    out: str = typer.Option("src/dataprep/artifacts", "--out", "-o"),
):
    """Run a full ingest + profile pipeline from a JSON config file.

    Config format:
    {
        "source": "postgresql",
        "connector": { "host": "...", "database": "...", "user": "...", "password": "..." },
        "query": { "table": "orders" },
        "ingest": { "batch_size": 10000 },
        "profile": { "mode": "full", "correlation": "pearson" }
    }
    """
    async def _run_pipeline():
        cfg = _load_config(config_file)
        source = cfg.get("source")
        if not source:
            console.print("[red]'source' is required in config[/red]")
            raise typer.Exit(1)

        connector_cfg: dict = cfg.get("connector", {})
        raw_query: dict = cfg.get("query", {})
        ingest_opts: dict = cfg.get("ingest", {})
        profile_opts: dict = cfg.get("profile", {})

        connector, query = _build_connector(
            source,
            connector_cfg,
            path=raw_query.get("path"),
            table=raw_query.get("table"),
            url=raw_query.get("url"),
            topic=raw_query.get("topic"),
            endpoint=raw_query.get("endpoint"),
        )

        ctx = _make_context(out)

        console.rule("[bold]Ingest")
        ingest_pipeline = IngestPipeline(IngestConfig(
            connector=connector,
            query=query,
            options=ingest_opts,
        ))
        result = ingest_pipeline.validate(ctx)
        if not result.ok:
            for err in result.errors:
                console.print(f"[red]{err}[/red]")
            raise typer.Exit(1)

        ingest_plan = ingest_pipeline.plan(ctx)
        output_artifact_id: str | None = None

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:
            task = p.add_task("Ingesting...", total=None)
            async for event in ingest_pipeline.execute(ingest_plan, ctx):
                p.update(task, description=event.message)
                if event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    output_artifact_id = payload.get("output_artifact_id")
                    console.print(f"[green][OK] {event.message}[/green]")
                elif event.event_type == EventType.ERROR:
                    console.print(f"[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

        await connector.close()

        if not output_artifact_id:
            console.print("[yellow]No artifact produced, skipping profile.[/yellow]")
            return

        console.rule("[bold]Profile")
        profile_pipeline = ProfilePipeline(ProfileOptions(
            mode=profile_opts.get("mode", "full"),
            sample_strategy=profile_opts.get("sample_strategy", "none"),
            sample_size=profile_opts.get("sample_size", 100_000),
            correlation_method=profile_opts.get("correlation", "pearson"),
        ))
        result = profile_pipeline.validate(ctx)
        if not result.ok:
            for err in result.errors:
                console.print(f"[red]{err}[/red]")
            raise typer.Exit(1)

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
                pct = int((event.progress_pct or 0) * 100)
                p.update(task, completed=pct, description=event.message)
                if event.event_type == EventType.DONE:
                    payload = event.payload or {}
                    console.print(f"[green][OK] {event.message}[/green]")
                    await _print_profile_table(ctx, payload.get("summary_artifact_id"))
                elif event.event_type == EventType.ERROR:
                    console.print(f"[red][ERROR] {event.message}[/red]")
                    raise typer.Exit(1)

    _run(_run_pipeline())


async def _print_profile_table(ctx: PipelineContext, summary_id: str | None) -> None:
    if not summary_id:
        return
    try:
        summary = await ctx.artifact_store.load_json(summary_id)
        t = Table(title="Column Summary", show_lines=True)
        t.add_column("Column", style="bold")
        t.add_column("Type")
        t.add_column("Null%", justify="right")
        t.add_column("Distinct", justify="right")
        t.add_column("Min")
        t.add_column("Max")
        t.add_column("Warnings", style="yellow")
        for col in summary.get("columns", []):
            t.add_row(
                col["name"],
                col["dtype"],
                f"{col['null_pct'] * 100:.1f}%",
                str(col["distinct_count"]),
                str(col.get("min", "")),
                str(col.get("max", "")),
                ", ".join(col.get("warnings", [])),
            )
        console.print(t)
    except Exception:
        pass


if __name__ == "__main__":
    app()