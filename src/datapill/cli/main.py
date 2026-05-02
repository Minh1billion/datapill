import typer
from importlib.metadata import version

app = typer.Typer(
    name="datapill",
    help="datapill - data ingestion and transformation pipelines",
    no_args_is_help=True,
)

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