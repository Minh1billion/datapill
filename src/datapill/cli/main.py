import typer

from .cmd_ingest import app as ingest_app
from .cmd_artifact import app as artifact_app

app = typer.Typer(
    name="datapill",
    help="datapill - data ingestion and transformation pipelines",
    no_args_is_help=True,
)

app.add_typer(ingest_app, name="ingest")
app.add_typer(artifact_app, name="artifact")


def main() -> None:
    app()