from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from datapill.connectors.base import BaseConnector
from datapill.connectors.kafka import KafkaConnector
from datapill.connectors.local_file import LocalFileConnector
from datapill.connectors.mysql import MySQLConnector
from datapill.connectors.postgresql import PostgreSQLConnector
from datapill.connectors.rest import RESTConnector
from datapill.connectors.s3 import S3Connector

from ._shared import SOURCES, console, db_exec, load_config, read_local, require_db_config, run_async, write_local

app = typer.Typer()


def _build_connector_for_action(source: str, config: dict, path: str | None, table: str | None, url: str | None, topic: str | None, endpoint: str | None) -> BaseConnector:
    if source == "local_file":
        return LocalFileConnector({"default_path": path or "."})
    if source == "postgresql":
        require_db_config(config, source)
        if table:
            config["default_table"] = table
        return PostgreSQLConnector(config)
    if source == "mysql":
        require_db_config(config, source)
        if table:
            config["default_table"] = table
        return MySQLConnector(config)
    if source == "s3":
        if url:
            config["default_url"] = url
        return S3Connector(config)
    if source == "rest":
        if endpoint:
            config["default_endpoint"] = endpoint
        return RESTConnector(config)
    if source == "kafka":
        if topic:
            config["default_topic"] = topic
        return KafkaConnector(config)
    console.print(f"[red]Unknown source: {source}. Available: {SOURCES}[/red]")
    raise typer.Exit(1)


@app.command("connector")
def cmd_connector(
    action: str = typer.Argument(..., help="test | schema | upload | download | list | exec | truncate | produce"),
    source: str = typer.Option(..., "--source", "-s", help=f"Connector type: {SOURCES}"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to JSON config file"),
    path: Optional[str] = typer.Option(None, "--path", help="Local file path"),
    table: Optional[str] = typer.Option(None, "--table", help="Table name (postgresql | mysql)"),
    url: Optional[str] = typer.Option(None, "--url", help="S3 URL (s3://bucket/key)"),
    topic: Optional[str] = typer.Option(None, "--topic", help="Kafka topic name"),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", help="REST endpoint"),
    src_path: Optional[str] = typer.Option(None, "--src-path", help="Source local file path (upload)"),
    dest_url: Optional[str] = typer.Option(None, "--dest-url", help="Destination S3 URL (upload)"),
    out_path: Optional[str] = typer.Option(None, "--out-path", help="Local output path (download)"),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="S3 key prefix filter (list)"),
    sql: Optional[str] = typer.Option(None, "--sql", help="Raw SQL to execute (exec)"),
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
    async def _exec():
        config = load_config(config_file)
        connector = _build_connector_for_action(source, config, path, table, url, topic, endpoint)

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
            df = read_local(src, fmt)
            if source == "s3":
                if not dest_url:
                    console.print("[red]--dest-url is required for s3 upload[/red]")
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
                console.print("[red]--url is required for download[/red]")
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
            write_local(df, dest, dest.suffix.lstrip(".").lower() or "parquet")
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
            await db_exec(source, config, sql)

        elif action == "truncate":
            if source not in ("postgresql", "mysql"):
                console.print("[red]truncate is only supported for postgresql and mysql[/red]")
                raise typer.Exit(1)
            if not table:
                console.print("[red]--table is required for truncate[/red]")
                raise typer.Exit(1)
            stmt = f'TRUNCATE TABLE "{table}"' if source == "postgresql" else f"TRUNCATE TABLE `{table}`"
            await db_exec(source, config, stmt)

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
            df = read_local(src, src.suffix.lstrip(".").lower())
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

    run_async(_exec())