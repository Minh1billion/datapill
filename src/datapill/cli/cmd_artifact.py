from typing import Optional

import typer
from rich.table import Table, box
from rich.text import Text

from ..storage.artifact_store import Artifact, ArtifactStore
from .shared import (
    BOLD_WHITE,
    ORANGE,
    GRAY,
    WHITE,
    GREEN,
    RED,
    YELLOW,
    CYAN,
    _rule,
    err,
    out,
    with_spinner,
)

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
        out.print("no artifacts found", style=GRAY)
        return

    t = Table(
        box=box.SIMPLE,
        show_edge=False,
        header_style=BOLD_WHITE,
        padding=(0, 2),
        pad_edge=False,
        border_style=GRAY,
    )
    t.add_column("run", style=f"bold {ORANGE}", no_wrap=True)
    t.add_column("pipeline", style=WHITE, no_wrap=True)
    t.add_column("when", style=GRAY, no_wrap=True)
    t.add_column("rows", justify="right", style=GRAY)
    t.add_column("sample", justify="center", style=YELLOW)
    t.add_column("mat", justify="center")

    for a in artifacts:
        _ss = a.options.get("sample_size") or a.sample_size
        rows_str = f"{int(_ss):,}" if a.is_sample and _ss else ""
        t.add_row(
            a.run_id,
            a.pipeline,
            _fmt_ts(a),
            rows_str,
            Text("y", style=YELLOW) if a.is_sample else "",
            Text("y", style=GREEN) if a.materialized else Text("·", style=GRAY),
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
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"artifact not found: {run_id}", style=RED))
        raise typer.Exit(1)

    t = Table(box=None, show_header=False, show_edge=False, padding=(0, 2), pad_edge=False)
    t.add_column(style=GRAY, no_wrap=True, min_width=14)
    t.add_column(no_wrap=True)
    t.add_row("run", Text(artifact.run_id, style=f"bold {ORANGE}"))
    t.add_row("pipeline", Text(artifact.pipeline, style=WHITE))
    t.add_row("timestamp", Text(_fmt_ts(artifact), style=GRAY))
    t.add_row(
        "sample",
        Text(f"yes  ({artifact.sample_size:,} rows)", style=YELLOW) if artifact.is_sample else Text("no", style=GRAY),
    )
    t.add_row(
        "materialized",
        Text("yes", style=GREEN) if artifact.materialized else Text("no", style=GRAY),
    )
    if artifact.path:
        t.add_row("path", Text(artifact.path, style=ORANGE))
    if artifact.parent_run_id:
        t.add_row("parent", Text(artifact.parent_run_id, style=GRAY))
    out.print(t)

    if artifact.schema:
        _rule("schema")
        st = Table(box=None, show_header=True, show_edge=False, header_style=BOLD_WHITE, padding=(0, 2), pad_edge=False)
        st.add_column("column", style=f"bold {CYAN}", no_wrap=True)
        st.add_column("type", style=GRAY, no_wrap=True)
        for col, dtype in artifact.schema.items():
            st.add_row(col, dtype)
        out.print(st)

    if artifact.options:
        _rule("options")
        ot = Table(box=None, show_header=True, show_edge=False, header_style=BOLD_WHITE, padding=(0, 2), pad_edge=False)
        ot.add_column("option", style=GRAY, no_wrap=True)
        ot.add_column("value", style=WHITE, no_wrap=True)
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
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"artifact not found: {run_id}", style=RED))
        raise typer.Exit(1)

    for i, a in enumerate(chain):
        indent = "  " * i
        connector = "└─ " if i > 0 else ""
        is_target = a.run_id == run_id
        out.print(
            Text(f"{indent}{connector}", style=GRAY)
            + Text(a.pipeline, style=f"bold {WHITE}" if is_target else GRAY)
            + Text(f"  {a.run_id}", style=ORANGE if is_target else GRAY)
            + Text(f"  {_fmt_ts(a)}", style=GRAY)
        )


@app.command("delete")
def delete_artifact(
    run_id: str = typer.Argument(help="run id of the artifact to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip confirmation"),
    cascade: bool = typer.Option(False, "--cascade", help="also delete all child artifacts"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    store = _open_store(store_path)
    artifact = store.get(run_id)

    if not artifact:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"artifact not found: {run_id}", style=RED))
        store.close()
        raise typer.Exit(1)

    if not yes:
        if cascade:
            subtree = store.lineage(run_id)
            out.print(Text(f"  will delete {len(subtree)} artifact(s):", style=WHITE))
            for a in reversed(subtree):
                out.print(
                    Text(f"    {a.run_id}", style=ORANGE)
                    + Text(f"  {a.pipeline}", style=GRAY)
                    + (Text("  materialized", style=GREEN) if a.materialized else Text(""))
                )
        else:
            out.print(
                Text("delete  ", style=GRAY)
                + Text(run_id, style=f"bold {WHITE}")
                + Text(f"  {artifact.pipeline}", style=GRAY)
                + (Text("  materialized", style=GREEN) if artifact.materialized else Text(""))
            )
        typer.confirm("confirm?", abort=True)

    with with_spinner(f"deleting {run_id}"):
        try:
            ok = store.delete(run_id, cascade=cascade)
        except RuntimeError as e:
            store.close()
            err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(str(e), style=RED))
            raise typer.Exit(1)
        store.close()

    if ok:
        out.print(Text("  [OK] deleted  ", style=f"bold {GREEN}") + Text(run_id, style=GRAY))
    else:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text("delete failed", style=RED))
        raise typer.Exit(1)


@app.command("purge")
def purge_artifacts(
    pipeline: Optional[str] = typer.Option(None, "--pipeline", "-p", help="limit to a specific pipeline"),
    keep: int = typer.Option(0, "--keep", "-k", help="number of most recent root artifacts to keep"),
    only_samples: bool = typer.Option(False, "--samples-only", help="only delete sample artifacts"),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip confirmation"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    store = _open_store(store_path)

    filters, params = ["parent_run_id IS NULL"], []
    if pipeline:
        filters.append("pipeline = ?")
        params.append(pipeline)
    where = "WHERE " + " AND ".join(filters)
    roots = store._db.execute(
        f"SELECT run_id FROM artifacts {where} ORDER BY timestamp DESC", params
    ).fetchall()
    roots_to_delete = [r[0] for r in roots][keep:]

    if not roots_to_delete:
        out.print("nothing to purge", style=GRAY)
        store.close()
        return

    out.print(
        Text(f"  {len(roots_to_delete)} root artifact(s) and their subtrees will be deleted", style=WHITE)
        + (Text(f"  keeping latest {keep}", style=GRAY) if keep else Text(""))
    )

    if not yes:
        typer.confirm("confirm?", abort=True)

    with with_spinner("purging"):
        deleted = store.purge(pipeline=pipeline, keep=keep, only_samples=only_samples)
        store.close()

    out.print(Text("  [OK] purged  ", style=f"bold {GREEN}") + Text(f"{len(deleted)} artifact(s)", style=WHITE))


@app.command("usage")
def disk_usage(
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    with with_spinner("calculating"):
        store = _open_store(store_path)
        total_bytes = store.disk_usage()
        count = len(store.list(limit=10_000))
        store.close()

    t = Table(box=None, show_header=False, show_edge=False, padding=(0, 2), pad_edge=False)
    t.add_column(style=GRAY, no_wrap=True, min_width=14)
    t.add_column(style=f"bold {WHITE}", no_wrap=True)
    t.add_row("artifacts", str(count))
    t.add_row("disk usage", _fmt_size(total_bytes))
    out.print(t)