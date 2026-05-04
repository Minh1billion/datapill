import asyncio
import json
from typing import Optional

import typer
from rich.table import Table, box
from rich.text import Text

from ..features.profile.pipeline import ProfilePipeline
from ..features.profile.stats import ProfileOptions
from ..core.context import Context
from ..storage.artifact_store import ArtifactStore
from .shared import (
    BOLD_WHITE,
    ORANGE,
    GRAY,
    WHITE,
    GREEN,
    RED,
    YELLOW,
    CYAN,
    MAGENTA,
    _rule,
    err,
    out,
    print_run_summary,
    print_schema,
    run_pipeline,
    with_spinner,
)

app = typer.Typer(help="profile a dataset artifact")

_SEVERITY_STYLE = {
    "error": RED,
    "warn": YELLOW,
    "info": CYAN,
}


@app.command("run")
def run_profile(
    parent_run_id: str = typer.Argument(help="run id of the parent artifact (ingest or preprocess)"),
    mode: str = typer.Option("full", "--mode", help="'full' computes histograms/correlations/patterns, 'summary' is faster"),
    sample_strategy: str = typer.Option("none", "--sample-strategy", help="sampling strategy: none | random | reservoir"),
    sample_size: int = typer.Option(100_000, "--sample-size", help="number of rows to use when sample-strategy != none"),
    chunk_size: int = typer.Option(100_000, "--chunk-size", help="rows per processing chunk"),
    correlation: str = typer.Option("pearson", "--correlation", help="correlation method: pearson | spearman | none"),
    correlation_threshold: float = typer.Option(0.3, "--correlation-threshold", help="only show pairs with |r| >= threshold"),
    no_patterns: bool = typer.Option(False, "--no-patterns", help="skip string pattern detection"),
    histogram_bins: int = typer.Option(20, "--histogram-bins", help="number of histogram bins for numeric columns"),
    cardinality_limit: int = typer.Option(100, "--cardinality-limit", help="max distinct values to show in top_values"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
    schema: bool = typer.Option(False, "--schema", help="print column schema after profiling"),
) -> None:
    options = ProfileOptions(
        mode=mode,
        chunk_size=chunk_size,
        sample_size=sample_size,
        sample_strategy=sample_strategy,
        histogram_bin_count=histogram_bins,
        cardinality_limit=cardinality_limit,
        detect_patterns=not no_patterns,
        correlation_method=correlation,
        correlation_threshold=correlation_threshold,
    )

    artifact_store = ArtifactStore(store_path)
    context = Context(artifact_store=artifact_store)
    pipeline = ProfilePipeline(parent_run_id=parent_run_id, options=options)

    validation = pipeline.validate(context)
    if not validation.ok:
        for e in validation.errors:
            err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(e, style=RED))
        raise typer.Exit(1)

    async def _run() -> None:
        plan = pipeline.plan(context)
        await run_pipeline(pipeline.execute(plan, context))

        if context.artifact:
            art = context.artifact
            print_run_summary({"run_id": art.run_id, "ref": art.ref})
            if schema and art.schema:
                print_schema(art.schema)
            if art.materialized and art.path:
                _rule("output")
                out.print(
                    Text("  profile: ", style=GRAY)
                    + Text(art.path, style=f"bold {ORANGE}")
                )
                out.print()

    asyncio.run(_run())


@app.command("show")
def show_profile(
    run_id: str = typer.Argument(help="run id of the profile artifact"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    with with_spinner(f"loading {run_id}"):
        store = ArtifactStore(store_path)
        artifact = store.get(run_id)
        store.close()

    if not artifact:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"artifact not found: {run_id}", style=RED))
        raise typer.Exit(1)

    if artifact.pipeline != "profile":
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"artifact {run_id!r} is not a profile (got {artifact.pipeline!r})", style=RED))
        raise typer.Exit(1)

    profile_result: dict | None = None
    if artifact.materialized and artifact.path:
        store = ArtifactStore(store_path)
        abs_path = store.path / artifact.path
        store.close()
        if abs_path.exists():
            profile_result = json.loads(abs_path.read_text())

    if profile_result is None and "profile_summary" in artifact.options:
        profile_result = artifact.options["profile_summary"]

    if profile_result is None:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text("no profile data found — artifact may have been created without materialization", style=RED))
        raise typer.Exit(1)

    summary = profile_result.get("summary", {})
    _rule("dataset")
    dt = Table(box=None, show_header=False, show_edge=False, padding=(0, 2), pad_edge=False)
    dt.add_column(style=GRAY, no_wrap=True, min_width=18)
    dt.add_column(style=WHITE, no_wrap=True)
    dt.add_row("run", Text(artifact.run_id, style=f"bold {ORANGE}"))
    dt.add_row("parent", Text(artifact.parent_run_id or "—", style=GRAY))
    dt.add_row("rows", f"{summary.get('n_rows', 0):,}")
    dt.add_row("columns", str(summary.get('n_columns', 0)))
    dt.add_row("memory", f"{summary.get('memory_mb', 0):.2f} MB")
    dt.add_row("null %", f"{summary.get('total_null_pct', 0) * 100:.2f}%")
    dt.add_row("duplicate %", f"{summary.get('duplicate_pct', 0) * 100:.2f}%")
    col_types = summary.get("column_types", {})
    dt.add_row(
        "column types",
        Text(f"{col_types.get('numeric', 0)} numeric  ", style=CYAN)
        + Text(f"{col_types.get('categorical', 0)} categorical  ", style=MAGENTA)
        + Text(f"{col_types.get('datetime', 0)} datetime", style=YELLOW),
    )
    if artifact.is_sample:
        dt.add_row("sampled", Text(f"yes  ({artifact.sample_size:,} rows)", style=YELLOW))
    out.print(dt)

    columns = profile_result.get("columns", [])
    if columns:
        _rule("columns")
        ct = Table(
            box=box.SIMPLE,
            show_edge=False,
            header_style=BOLD_WHITE,
            padding=(0, 2),
            pad_edge=False,
            border_style=GRAY,
        )
        ct.add_column("column", style=f"bold {CYAN}", no_wrap=True)
        ct.add_column("dtype", style=GRAY, no_wrap=True)
        ct.add_column("null %", justify="right", style=GRAY)
        ct.add_column("distinct", justify="right", style=GRAY)
        ct.add_column("mean / top value", style=WHITE)
        ct.add_column("warns", justify="center")

        for cp in columns:
            null_pct = cp.get("null_pct", 0)
            null_str = f"{null_pct * 100:.1f}%"
            null_style = RED if null_pct > 0.3 else (YELLOW if null_pct > 0.1 else GRAY)

            mean_or_top = ""
            if cp.get("mean") is not None:
                try:
                    mean_or_top = f"{float(cp['mean']):.4g}"
                except (TypeError, ValueError):
                    mean_or_top = str(cp["mean"])
            elif cp.get("top_values"):
                top = cp["top_values"]
                if top:
                    mean_or_top = str(top[0].get("value", ""))[:30]

            col_warnings = cp.get("warnings", [])
            warn_count = len(col_warnings)
            warn_cell = (
                Text(str(warn_count), style=RED) if any(w.get("severity") == "error" for w in col_warnings)
                else Text(str(warn_count), style=YELLOW) if warn_count > 0
                else Text("·", style=GRAY)
            )

            ct.add_row(
                cp.get("name", ""),
                cp.get("dtype_inferred", cp.get("dtype_physical", "")),
                Text(null_str, style=null_style),
                str(cp.get("distinct_count", "")),
                mean_or_top,
                warn_cell,
            )
        out.print(ct)

    correlations = profile_result.get("correlations", [])
    if correlations:
        _rule("correlations")
        cort = Table(
            box=box.SIMPLE,
            show_edge=False,
            header_style=BOLD_WHITE,
            padding=(0, 2),
            pad_edge=False,
            border_style=GRAY,
        )
        cort.add_column("col a", style=CYAN, no_wrap=True)
        cort.add_column("col b", style=CYAN, no_wrap=True)
        cort.add_column("method", style=GRAY, no_wrap=True)
        cort.add_column("r", justify="right", style=WHITE)

        for pair in sorted(correlations, key=lambda x: abs(x.get("value", 0)), reverse=True):
            val = pair.get("value", 0)
            val_style = RED if abs(val) >= 0.8 else (YELLOW if abs(val) >= 0.5 else WHITE)
            cort.add_row(
                pair.get("col_a", ""),
                pair.get("col_b", ""),
                pair.get("method", ""),
                Text(f"{val:+.4f}", style=val_style),
            )
        out.print(cort)

    warnings = profile_result.get("warnings", [])
    if warnings:
        _rule("warnings")
        for sev in ("error", "warn", "info"):
            for w in [x for x in warnings if x.get("severity") == sev]:
                col = w.get("column", "")
                col_part = Text(f"  {col:<24}", style=CYAN) if col and col != "__dataset__" else Text(f"  {'(dataset)':<24}", style=GRAY)
                out.print(
                    Text(f"  [{sev.upper():<5}] ", style=f"bold {_SEVERITY_STYLE.get(sev, GRAY)}")
                    + col_part
                    + Text(w.get("code", ""), style=_SEVERITY_STYLE.get(sev, GRAY))
                )

    _rule()


@app.command("warnings")
def show_warnings(
    run_id: str = typer.Argument(help="run id of the profile artifact"),
    severity: Optional[str] = typer.Option(None, "--severity", help="filter by severity: error | warn | info"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    with with_spinner(f"loading {run_id}"):
        store = ArtifactStore(store_path)
        artifact = store.get(run_id)
        store.close()

    if not artifact or artifact.pipeline != "profile":
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"profile artifact not found: {run_id}", style=RED))
        raise typer.Exit(1)

    profile_result: dict | None = None
    if artifact.materialized and artifact.path:
        store = ArtifactStore(store_path)
        abs_path = store.path / artifact.path
        store.close()
        if abs_path.exists():
            profile_result = json.loads(abs_path.read_text())

    if profile_result is None and "profile_summary" in artifact.options:
        profile_result = artifact.options["profile_summary"]

    if profile_result is None:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text("no profile data found", style=RED))
        raise typer.Exit(1)

    warnings = profile_result.get("warnings", [])
    if severity:
        warnings = [w for w in warnings if w.get("severity") == severity]

    if not warnings:
        out.print(Text("  no warnings", style=GRAY) + (Text(f" (severity={severity})", style=GRAY) if severity else Text("")))
        return

    for sev in ("error", "warn", "info"):
        for w in [x for x in warnings if x.get("severity") == sev]:
            col = w.get("column", "")
            col_part = Text(f"  {col:<24}", style=CYAN) if col and col != "__dataset__" else Text(f"  {'(dataset)':<24}", style=GRAY)
            out.print(
                Text(f"  [{sev.upper():<5}] ", style=f"bold {_SEVERITY_STYLE.get(sev, GRAY)}")
                + col_part
                + Text(w.get("code", ""), style=_SEVERITY_STYLE.get(sev, GRAY))
            )