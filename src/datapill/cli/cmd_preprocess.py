import asyncio
import json
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich.table import Table, box
from rich.text import Text

from ..core.context import Context
from ..features.preprocess.ops import validate_op
from ..features.preprocess.pipeline import PreprocessPipeline
from ..utils.loader import load_dataframe
from ..storage.artifact_store import ArtifactStore
from .shared import (
    BOLD_WHITE,
    CYAN,
    GRAY,
    GREEN,
    ORANGE,
    RED,
    WHITE,
    YELLOW,
    _rule,
    err,
    out,
    print_run_summary,
    print_schema,
    run_pipeline,
    with_spinner,
)

app = typer.Typer(help="preprocess a dataset artifact")


def _parse_op(raw: str) -> dict:
    parts = raw.split(":")
    head = parts[0]
    if "." not in head:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"op must be group.type, got: {head!r}", style=RED))
        raise typer.Exit(1)
    group, type_ = head.split(".", 1)
    op: dict = {"group": group, "type": type_}
    for part in parts[1:]:
        if "=" not in part:
            err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"op param must be key=value, got: {part!r}", style=RED))
            raise typer.Exit(1)
        key, val = part.split("=", 1)
        if "," in val:
            op[key] = val.split(",")
        elif val.lstrip("-").isdigit():
            op[key] = int(val)
        elif val.replace(".", "", 1).lstrip("-").isdigit():
            op[key] = float(val)
        elif val.lower() in ("true", "false"):
            op[key] = val.lower() == "true"
        else:
            op[key] = val
    return op


def _load_ops_file(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"ops file not found: {p}", style=RED))
        raise typer.Exit(1)
    if p.suffix != ".json":
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"ops file must be .json: {p}", style=RED))
        raise typer.Exit(1)
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"invalid ops file: {exc}", style=RED))
        raise typer.Exit(1)
    if not isinstance(data, list):
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text("ops file must contain a JSON array", style=RED))
        raise typer.Exit(1)
    return data


def _print_dataframe(df: pl.DataFrame, n: int) -> None:
    sample = df.head(n)
    t = Table(
        box=box.SIMPLE,
        show_edge=False,
        header_style=BOLD_WHITE,
        padding=(0, 2),
        pad_edge=False,
        border_style=GRAY,
    )
    for col in sample.columns:
        t.add_column(col, style=CYAN, no_wrap=True)
    for row in sample.iter_rows():
        t.add_row(*[str(v) if v is not None else Text("null", style=GRAY) for v in row])
    out.print(t)
    out.print(Text(f"  {n} of {len(df):,} rows", style=GRAY))


@app.command("run")
def run_preprocess(
    parent_run_id: str = typer.Argument(help="run id of the parent artifact (ingest or preprocess)"),
    op: Optional[list[str]] = typer.Option(None, "--op", help="op in format group.type:key=value:key=value"),
    ops: Optional[str] = typer.Option(None, "--ops", help="path to ops .json file"),
    materialized: bool = typer.Option(False, "--materialize", "-m", help="apply ops and write output to parquet"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
    schema: bool = typer.Option(False, "--schema", help="print column schema after run"),
) -> None:
    if not op and not ops:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text("provide --op or --ops", style=RED))
        raise typer.Exit(1)
    if op and ops:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text("use --op or --ops, not both", style=RED))
        raise typer.Exit(1)

    ops_list: list[dict] = _load_ops_file(ops) if ops else [_parse_op(o) for o in op]

    for i, o in enumerate(ops_list):
        op_errors = validate_op(o)
        for e in op_errors:
            err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"ops[{i}]: {e}", style=RED))
        if op_errors:
            raise typer.Exit(1)

    artifact_store = ArtifactStore(store_path)
    context = Context(artifact_store=artifact_store)
    pipeline = PreprocessPipeline(
        parent_run_id=parent_run_id,
        ops=ops_list,
        options={"materialized": materialized},
    )

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
                out.print(Text("  parquet: ", style=GRAY) + Text(art.path, style=f"bold {ORANGE}"))
                out.print()

    asyncio.run(_run())


@app.command("materialize")
def materialize_preprocess(
    run_id: str = typer.Argument(help="run id of an existing preprocess artifact"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
    schema: bool = typer.Option(False, "--schema", help="print column schema after materialize"),
) -> None:
    with with_spinner(f"loading {run_id}"):
        store = ArtifactStore(store_path)
        artifact = store.get(run_id)
        store.close()

    if not artifact:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"artifact not found: {run_id}", style=RED))
        raise typer.Exit(1)

    if artifact.pipeline != "preprocess":
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"artifact {run_id!r} is not a preprocess artifact (got {artifact.pipeline!r})", style=RED))
        raise typer.Exit(1)

    if artifact.materialized:
        out.print(Text("  already materialized: ", style=GRAY) + Text(artifact.path or "", style=f"bold {ORANGE}"))
        raise typer.Exit(0)

    ops_list: list[dict] = artifact.options.get("ops", [])
    if not ops_list:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text("artifact has no ops config", style=RED))
        raise typer.Exit(1)

    artifact_store = ArtifactStore(store_path)
    context = Context(artifact_store=artifact_store)
    pipeline = PreprocessPipeline(
        parent_run_id=artifact.parent_run_id,
        ops=ops_list,
        options={"materialized": True},
    )

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
                out.print(Text("  parquet: ", style=GRAY) + Text(art.path, style=f"bold {ORANGE}"))
                out.print()

    asyncio.run(_run())


@app.command("preview")
def preview_preprocess(
    run_id: str = typer.Argument(help="run id of the preprocess artifact"),
    rows: int = typer.Option(20, "--rows", "-n", help="number of rows to show"),
    store_path: str = typer.Option(".datapill", "--store", help="artifact store directory"),
) -> None:
    with with_spinner(f"loading {run_id}"):
        store = ArtifactStore(store_path)
        artifact = store.get(run_id)
        store.close()

    if not artifact:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"artifact not found: {run_id}", style=RED))
        raise typer.Exit(1)

    if artifact.pipeline != "preprocess":
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"artifact {run_id!r} is not a preprocess artifact", style=RED))
        raise typer.Exit(1)

    if artifact.materialized and artifact.path:
        with with_spinner("reading parquet"):
            store = ArtifactStore(store_path)
            abs_path = store.path / artifact.path
            store.close()
            df = pl.read_parquet(abs_path)
        _rule("preview")
        _print_dataframe(df, rows)
        return

    ops_list: list[dict] = artifact.options.get("ops", [])
    if not ops_list:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text("artifact has no ops config to apply", style=RED))
        raise typer.Exit(1)

    artifact_store = ArtifactStore(store_path)
    context = Context(artifact_store=artifact_store)

    async def _preview() -> None:
        parent = context.artifact_store.get(artifact.parent_run_id)
        if not parent:
            err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"parent artifact not found: {artifact.parent_run_id}", style=RED))
            raise typer.Exit(1)

        with with_spinner("loading sample data"):
            df = await load_dataframe(parent, context)

        df = df.head(rows * 10)

        from ..features.preprocess.ops import apply_op
        for op in ops_list:
            if op.get("type") == "join":
                out.print(Text("  [SKIP] ", style=YELLOW) + Text("join ops skipped in preview", style=GRAY))
                continue
            try:
                df = apply_op(df, op)
            except Exception as exc:
                err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"{op.get('group')}.{op.get('type')}: {exc}", style=RED))
                raise typer.Exit(1)

        _rule("preview")
        _print_dataframe(df, rows)

    asyncio.run(_preview())


@app.command("ops")
def list_ops(
    group: Optional[str] = typer.Option(None, "--group", "-g", help="filter by group"),
) -> None:
    from ..features.preprocess.ops import _DISPATCH, _REQUIRED

    groups = [group] if group else list(_DISPATCH.keys())
    unknown = [g for g in groups if g not in _DISPATCH]
    if unknown:
        err.print(Text("[FAIL] ", style=f"bold {RED}") + Text(f"unknown group(s): {', '.join(unknown)}", style=RED))
        raise typer.Exit(1)

    for g in groups:
        _rule(g)
        t = Table(box=None, show_header=True, show_edge=False, header_style=BOLD_WHITE, padding=(0, 2), pad_edge=False)
        t.add_column("op", style=f"bold {CYAN}", no_wrap=True)
        t.add_column("required params", style=GRAY)
        t.add_column("example", style=WHITE)

        examples = {
            ("schema", "cast"): "schema.cast:col=age:dtype=Int32",
            ("schema", "rename"): "schema.rename:old_name=new_name",
            ("schema", "drop_columns"): "schema.drop_columns:cols=a,b,c",
            ("schema", "select_columns"): "schema.select_columns:cols=id,name,age",
            ("schema", "reorder_columns"): "schema.reorder_columns:cols=id,name",
            ("clean", "fill_null"): "clean.fill_null:cols=salary:value=0",
            ("clean", "drop_null"): "clean.drop_null:cols=email,name",
            ("clean", "impute"): "clean.impute:cols=age,salary:strategy=median",
            ("clean", "clip"): "clean.clip:cols=score:min=0:max=100",
            ("clean", "winsorize"): "clean.winsorize:cols=revenue:lower=0.05:upper=0.95",
            ("clean", "drop_outlier"): "clean.drop_outlier:cols=price:method=iqr",
            ("clean", "flag_outlier"): "clean.flag_outlier:cols=price:method=zscore",
            ("clean", "replace_value"): "clean.replace_value:cols=status:mapping=...",
            ("transform", "normalize"): "transform.normalize:cols=age,salary",
            ("transform", "standardize"): "transform.standardize:cols=revenue",
            ("transform", "log_transform"): "transform.log_transform:cols=price:base=log10",
            ("transform", "bin"): "transform.bin:col=age:breaks=0,18,35,60,100",
            ("transform", "rank"): "transform.rank:cols=score:method=dense",
            ("transform", "power_transform"): "transform.power_transform:cols=income:method=sqrt",
            ("transform", "encode"): "transform.encode:col=city:method=onehot",
            ("transform", "math_expr"): "transform.math_expr:out_col=ratio:expr=...",
            ("parse", "trim"): "parse.trim:cols=name,email",
            ("parse", "lower"): "parse.lower:cols=email",
            ("parse", "upper"): "parse.upper:cols=country_code",
            ("parse", "regex_extract"): "parse.regex_extract:col=text:pattern=...:group=1",
            ("parse", "split_col"): "parse.split_col:col=full_name:sep= :out_cols=first,last",
            ("parse", "parse_datetime"): "parse.parse_datetime:cols=created_at:format=%Y-%m-%d",
            ("parse", "extract_datetime_part"): "parse.extract_datetime_part:col=ts:parts=year,month,dow",
            ("parse", "date_diff"): "parse.date_diff:col_a=end_date:col_b=start_date:unit=day",
            ("reshape", "filter_rows"): "reshape.filter_rows:expr=...",
            ("reshape", "dedup"): "reshape.dedup:cols=email:keep=first",
            ("reshape", "sort"): "reshape.sort:cols=created_at:descending=true",
            ("reshape", "add_column"): "reshape.add_column:col=source:value=web",
            ("reshape", "explode"): "reshape.explode:col=tags",
            ("reshape", "pivot"): "reshape.pivot:on=month:index=store_id:values=revenue",
            ("reshape", "unpivot"): "reshape.unpivot:on=jan,feb,mar:index=store_id",
            ("compose", "window_agg"): "compose.window_agg:col=revenue:fn=rolling_mean:window=7:partition_by=store_id",
            ("compose", "group_agg"): "compose.group_agg:by=region:aggs=...",
            ("compose", "feature_cross"): "compose.feature_cross:col_a=gender:col_b=age_group",
            ("compose", "resample"): "compose.resample:time_col=ts:every=1d:agg_col=revenue:fn=sum",
            ("compose", "sample"): "compose.sample:n=1000",
            ("compose", "custom_expr"): "compose.custom_expr:out_col=flag:expr=...",
        }

        for type_ in _DISPATCH[g]:
            required = _REQUIRED[g][type_]
            req_str = ", ".join(required) if required else Text("—", style=GRAY)
            example = examples.get((g, type_), "")
            t.add_row(type_, req_str, example)

        out.print(t)