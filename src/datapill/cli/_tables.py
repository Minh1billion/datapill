from rich.table import Table

from datapill.core.context import PipelineContext

from ._shared import console


async def print_profile_table(ctx: PipelineContext, summary_id: str | None) -> None:
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


async def print_classify_table(ctx: PipelineContext, artifact_id: str | None) -> None:
    if not artifact_id:
        return
    try:
        data = await ctx.artifact_store.load_json(artifact_id)
        t = Table(title="Column Classifications", show_lines=True)
        t.add_column("Column", style="bold")
        t.add_column("Semantic Type", style="cyan")
        t.add_column("Confidence", justify="right")
        t.add_column("Source")
        t.add_column("Overridden")
        t.add_column("Reasoning", style="dim")
        for col in data.get("columns", []):
            conf = col["confidence"]
            conf_color = "green" if conf >= 0.70 else "yellow" if conf >= 0.55 else "red"
            t.add_row(
                col["name"],
                col["semantic_type"],
                f"[{conf_color}]{conf:.2f}[/{conf_color}]",
                col["source"],
                "[yellow]yes[/yellow]" if col["overridden"] else "no",
                col["reasoning"],
            )
        console.print(t)
    except Exception:
        pass