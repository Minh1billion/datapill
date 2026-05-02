import typer
from importlib.metadata import version

from .cmd_artifact import app as artifact_app
from .cmd_ingest import app as ingest_app
from .cmd_profile import app as profile_app

app = typer.Typer(
    name="datapill",
    help="datapill - data ingestion and transformation pipelines",
    no_args_is_help=True,
)

app.add_typer(ingest_app, name="ingest")
app.add_typer(artifact_app, name="artifact")
app.add_typer(profile_app, name="profile")

def version_callback(value: bool):
    if value:
        print(f"datapill {version('datapill')}")
        raise typer.Exit()

@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    )
):
    pass