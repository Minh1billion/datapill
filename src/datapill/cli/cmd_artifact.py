from typing import Optional

import typer
from rich.text import Text

from ..storage.artifact_store import ArtifactStore, Artifact
from .shared import out, err

app = typer.Typer(help="inspect and manage pipeline artifacts")


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_ts(artifact: Artifact) -> str:
    return artifact.timestamp.strftime("%Y-%m-%d %H:%M")


def _open_store(store_path: str) -> ArtifactStore:
    return ArtifactStore(store_path)


@app.command("list")
def list_artifacts(
    pipeline: Optional[str] = typer.Option(None, "--pipeline", "-p", help="filter by pipeline name"),
    limit: int = typer.Option(20, "--limit", "-n", help="max number of artifacts to show"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    from rich.table import Table, box

    store = _open_store(store_path)
    artifacts = store.list(pipeline=pipeline, limit=limit)
    store.close()

    if not artifacts:
        out.print("no artifacts found", style="dim")
        return

    t = Table(box=box.SIMPLE_HEAD, show_edge=False, header_style="bold dim")
    t.add_column("run id", no_wrap=True)
    t.add_column("pipeline", no_wrap=True)
    t.add_column("when", style="dim", no_wrap=True)
    t.add_column("rows", justify="right", style="dim")
    t.add_column("sample", justify="center")
    t.add_column("materialized", justify="center")

    for a in artifacts:
        rows_str = str(a.options.get("sample_size", "")) if a.is_sample else ""
        t.add_row(
            a.run_id,
            a.pipeline,
            _fmt_ts(a),
            rows_str,
            "y" if a.is_sample else "",
            Text("y", style="green") if a.materialized else "",
        )

    out.print(t)


@app.command("show")
def show_artifact(
    run_id: str = typer.Argument(help="run id of the artifact"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    from rich.table import Table, box

    store = _open_store(store_path)
    artifact = store.get(run_id)
    store.close()

    if not artifact:
        err.print(f"x  artifact not found: {run_id}", style="red")
        raise typer.Exit(1)

    t = Table(box=None, show_header=False, padding=(0, 2, 0, 0), show_edge=False)
    t.add_column(style="dim", no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_row("run id",     Text(artifact.run_id, style="bold"))
    t.add_row("pipeline",   artifact.pipeline)
    t.add_row("timestamp",  _fmt_ts(artifact))
    t.add_row("sample",     f"yes  ({artifact.sample_size:,} rows)" if artifact.is_sample else "no")
    t.add_row("materialized", Text("yes", style="green") if artifact.materialized else "no")
    if artifact.path:
        t.add_row("path", artifact.path)
    if artifact.parent_run_id:
        t.add_row("parent", artifact.parent_run_id)
    out.print(t)

    if artifact.schema:
        out.print()
        schema_t = Table(box=box.SIMPLE_HEAD, show_edge=False, header_style="bold dim")
        schema_t.add_column("column", style="cyan", no_wrap=True)
        schema_t.add_column("type", style="dim", no_wrap=True)
        for col, dtype in artifact.schema.items():
            schema_t.add_row(col, dtype)
        out.print(schema_t)

    if artifact.options:
        out.print()
        opts_t = Table(box=box.SIMPLE_HEAD, show_edge=False, header_style="bold dim")
        opts_t.add_column("option", style="dim", no_wrap=True)
        opts_t.add_column("value", no_wrap=True)
        for k, v in artifact.options.items():
            opts_t.add_row(k, str(v))
        out.print(opts_t)


@app.command("lineage")
def show_lineage(
    run_id: str = typer.Argument(help="run id to trace lineage from"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    store = _open_store(store_path)
    chain = store.lineage(run_id)
    store.close()

    if not chain:
        err.print(f"x  artifact not found: {run_id}", style="red")
        raise typer.Exit(1)

    for i, a in enumerate(chain):
        prefix = "  " * i
        connector = "└─ " if i > 0 else ""
        label = Text(f"{prefix}{connector}{a.pipeline}", style="bold" if a.run_id == run_id else "")
        out.print(label + Text(f"  {a.run_id}", style="dim") + Text(f"  {_fmt_ts(a)}", style="dim"))


@app.command("delete")
def delete_artifact(
    run_id: str = typer.Argument(help="run id of the artifact to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip confirmation"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    store = _open_store(store_path)
    artifact = store.get(run_id)

    if not artifact:
        err.print(f"x  artifact not found: {run_id}", style="red")
        store.close()
        raise typer.Exit(1)

    if not yes:
        out.print(f"delete [bold]{run_id}[/bold]  pipeline=[dim]{artifact.pipeline}[/dim]  materialized=[dim]{artifact.materialized}[/dim]")
        typer.confirm("confirm?", abort=True)

    ok = store.delete(run_id)
    store.close()

    if ok:
        out.print(Text("  ✓ ", style="bold green") + Text(f"deleted {run_id}"))
    else:
        err.print("x  delete failed", style="red")
        raise typer.Exit(1)


@app.command("purge")
def purge_artifacts(
    pipeline: Optional[str] = typer.Option(None, "--pipeline", "-p", help="limit to a specific pipeline"),
    keep: int = typer.Option(0, "--keep", "-k", help="number of most recent artifacts to keep"),
    only_samples: bool = typer.Option(False, "--samples-only", help="only delete sample artifacts"),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip confirmation"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    store = _open_store(store_path)
    artifacts = store.list(pipeline=pipeline)

    candidates = artifacts[keep:]
    if only_samples:
        candidates = [a for a in candidates if a.is_sample]

    if not candidates:
        out.print("nothing to purge", style="dim")
        store.close()
        return

    out.print(f"[dim]will delete[/dim] [bold]{len(candidates)}[/bold] [dim]artifact(s)[/dim]" + (f"  keeping latest {keep}" if keep else ""))

    if not yes:
        typer.confirm("confirm?", abort=True)

    deleted = store.purge(pipeline=pipeline, keep=keep, only_samples=only_samples)
    store.close()

    out.print(Text("  ✓ ", style="bold green") + Text(f"purged {len(deleted)} artifact(s)"))


@app.command("usage")
def disk_usage(
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    store = _open_store(store_path)
    total_bytes = store.disk_usage()
    count = len(store.list(limit=10_000))
    store.close()

    out.print(f"  artifacts   [bold]{count}[/bold]")
    out.print(f"  disk usage  [bold]{_fmt_size(total_bytes)}[/bold]")