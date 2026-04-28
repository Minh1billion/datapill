import typer

from .cmd_classify import app as classify_app
from .cmd_connector import app as connector_app
from .cmd_export import app as export_app
from .cmd_ingest import app as ingest_app
from .cmd_list import app as list_app
from .cmd_pipeline import app as pipeline_app
from .cmd_preprocess import app as preprocess_app
from .cmd_profile import app as profile_app
from .cmd_run import app as run_app

app = typer.Typer(name="dp", help="datapill CLI", no_args_is_help=True)

app.registered_commands += ingest_app.registered_commands
app.registered_commands += profile_app.registered_commands
app.registered_commands += preprocess_app.registered_commands
app.registered_commands += classify_app.registered_commands
app.registered_commands += export_app.registered_commands
app.registered_commands += list_app.registered_commands
app.registered_commands += connector_app.registered_commands
app.registered_commands += run_app.registered_commands
app.add_typer(pipeline_app, name="pipeline")

if __name__ == "__main__":
    app()