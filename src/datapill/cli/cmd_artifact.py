from typing import Optional

import typer
from rich.table import Table, box
from rich.text import Text

from ..storage.artifact_store import Artifact, ArtifactStore
from .shared import C_ACCENT, C_ERR, C_HEAD, C_ID, C_MUTED, C_OK, C_PATH, C_REF, C_TYPE, C_VAL, C_WARN, _rule, err, out, with_spinner

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
    with with_spinner("loading artifacts"):
        store = _open_store(store_path)
        artifacts = store.list(pipeline=pipeline, limit=limit)
        store.close()

    if not artifacts:
        out.print("no artifacts found", style=C_MUTED)
        return

    t = Table(box=box.SIMPLE_HEAD, show_edge=False, header_style=C_HEAD, padding=(0, 1), pad_edge=False)
    t.add_column("run", style=f"bold {C_ID}", no_wrap=True)
    t.add_column("pipeline", style=C_VAL, no_wrap=True)
    t.add_column("when", style=C_MUTED, no_wrap=True)
    t.add_column("rows", justify="right", style=C_TYPE)
    t.add_column("sample", justify="center", style=C_WARN)
    t.add_column("mat", justify="center")

    for a in artifacts:
        rows_str = f"{a.options.get('sample_size', ''):,}" if a.is_sample else ""
        t.add_row(
            a.run_id,
            a.pipeline,
            _fmt_ts(a),
            rows_str,
            Text("y", style=C_WARN) if a.is_sample else "",
            Text("y", style=C_OK) if a.materialized else Text("·", style=C_MUTED),
        )

    out.print(t)


@app.command("show")
def show_artifact(
    run_id: str = typer.Argument(help="run id of the artifact"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    with with_spinner(f"loading {run_id}"):
        store = _open_store(store_path)
        artifact = store.get(run_id)
        store.close()

    if not artifact:
        err.print(Text("✖ ", style=f"bold {C_ERR}") + Text(f"artifact not found: {run_id}", style=C_ERR))
        raise typer.Exit(1)

    t = Table(box=None, show_header=False, show_edge=False, padding=(0, 1), pad_edge=False)
    t.add_column(style=C_MUTED, no_wrap=True, min_width=12)
    t.add_column(no_wrap=True)
    t.add_row("run",          Text(artifact.run_id, style=f"bold {C_ID}"))
    t.add_row("pipeline",     Text(artifact.pipeline, style=C_VAL))
    t.add_row("timestamp",    Text(_fmt_ts(artifact), style=C_MUTED))
    t.add_row("sample",       Text(f"yes  ({artifact.sample_size:,} rows)", style=C_WARN) if artifact.is_sample else Text("no", style=C_MUTED))
    t.add_row("materialized", Text("yes", style=C_OK) if artifact.materialized else Text("no", style=C_MUTED))
    if artifact.path:
        t.add_row("path",     Text(artifact.path, style=C_PATH))
    if artifact.parent_run_id:
        t.add_row("parent",   Text(artifact.parent_run_id, style=C_REF))
    out.print(t)

    if artifact.schema:
        _rule("schema")
        st = Table(box=None, show_header=True, show_edge=False, header_style=C_HEAD, padding=(0, 1), pad_edge=False)
        st.add_column("column", style=f"bold {C_ACCENT}", no_wrap=True)
        st.add_column("type", style=C_TYPE, no_wrap=True)
        for col, dtype in artifact.schema.items():
            st.add_row(col, dtype)
        out.print(st)

    if artifact.options:
        _rule("options")
        ot = Table(box=None, show_header=True, show_edge=False, header_style=C_HEAD, padding=(0, 1), pad_edge=False)
        ot.add_column("option", style=C_MUTED, no_wrap=True)
        ot.add_column("value", style=C_VAL, no_wrap=True)
        for k, v in artifact.options.items():
            ot.add_row(k, str(v))
        out.print(ot)

    _rule()


@app.command("lineage")
def show_lineage(
    run_id: str = typer.Argument(help="run id to trace lineage from"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    with with_spinner(f"tracing {run_id}"):
        store = _open_store(store_path)
        chain = store.lineage(run_id)
        store.close()

    if not chain:
        err.print(Text("✖ ", style=f"bold {C_ERR}") + Text(f"artifact not found: {run_id}", style=C_ERR))
        raise typer.Exit(1)

    for i, a in enumerate(chain):
        indent = "  " * i
        connector = "└─ " if i > 0 else ""
        is_target = a.run_id == run_id
        out.print(
            Text(f"{indent}{connector}", style=C_MUTED)
            + Text(a.pipeline, style=f"bold {C_VAL}" if is_target else C_MUTED)
            + Text(f"  {a.run_id}", style=C_ID if is_target else C_MUTED)
            + Text(f"  {_fmt_ts(a)}", style=C_MUTED)
        )


@app.command("delete")
def delete_artifact(
    run_id: str = typer.Argument(help="run id of the artifact to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip confirmation"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    store = _open_store(store_path)
    artifact = store.get(run_id)

    if not artifact:
        err.print(Text("✖ ", style=f"bold {C_ERR}") + Text(f"artifact not found: {run_id}", style=C_ERR))
        store.close()
        raise typer.Exit(1)

    if not yes:
        out.print(
            Text("delete  ", style=C_MUTED)
            + Text(run_id, style=f"bold {C_VAL}")
            + Text(f"  {artifact.pipeline}", style=C_MUTED)
            + (Text("  materialized", style=C_OK) if artifact.materialized else Text(""))
        )
        typer.confirm("confirm?", abort=True)

    with with_spinner(f"deleting {run_id}"):
        ok = store.delete(run_id)
        store.close()

    if ok:
        out.print(Text("  ✔ deleted  ", style=f"bold {C_OK}") + Text(run_id, style=C_MUTED))
    else:
        err.print(Text("✖ ", style=f"bold {C_ERR}") + Text("delete failed", style=C_ERR))
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
        out.print("nothing to purge", style=C_MUTED)
        store.close()
        return

    out.print(
        Text(f"  {len(candidates)} artifact(s) will be deleted", style=C_VAL)
        + (Text(f"  keeping latest {keep}", style=C_MUTED) if keep else Text(""))
    )

    if not yes:
        typer.confirm("confirm?", abort=True)

    with with_spinner("purging"):
        deleted = store.purge(pipeline=pipeline, keep=keep, only_samples=only_samples)
        store.close()

    out.print(Text("  ✔ purged  ", style=f"bold {C_OK}") + Text(f"{len(deleted)} artifact(s)", style=C_VAL))


@app.command("usage")
def disk_usage(
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    with with_spinner("calculating"):
        store = _open_store(store_path)
        total_bytes = store.disk_usage()
        count = len(store.list(limit=10_000))
        store.close()

    t = Table(box=None, show_header=False, show_edge=False, padding=(0, 1), pad_edge=False)
    t.add_column(style=C_MUTED, no_wrap=True, min_width=12)
    t.add_column(style=f"bold {C_VAL}", no_wrap=True)
    t.add_row("artifacts", str(count))
    t.add_row("disk usage", _fmt_size(total_bytes))
    out.print(t)