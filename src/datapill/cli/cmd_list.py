import typer
from rich.table import Table

from ._shared import console, make_context

app = typer.Typer()


@app.command("list")
def cmd_list(
    feature: str = typer.Option("", "--feature", "-f", help="Filter by feature: ingest | profile | preprocess"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max number of artifacts to show"),
):
    """List artifacts in the artifact store."""
    ctx = make_context()
    artifacts = ctx.artifact_store.list_artifacts()

    if feature:
        artifacts = [a for a in artifacts if a["feature"] == feature]
    artifacts = artifacts[:limit]

    if not artifacts:
        console.print("[yellow]No artifacts found[/yellow]")
        return

    t = Table(title="Artifacts", show_lines=True)
    t.add_column("Artifact ID", style="bold cyan")
    t.add_column("Run ID")
    t.add_column("Feature")
    t.add_column("Ext")
    t.add_column("Created At")

    for a in artifacts:
        t.add_row(a["artifact_id"], a["run_id"], a["feature"], a["ext"], a["created_at"][:19])

    console.print(t)