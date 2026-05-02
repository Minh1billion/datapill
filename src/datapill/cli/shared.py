import sys
from contextlib import contextmanager
from typing import Any, AsyncIterator, Callable, Generator

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from ..core.events import EventType, ProgressEvent

out = Console()
err = Console(stderr=True)

C_PRIMARY = "cyan"
C_SUCCESS = "green"
C_WARNING = "yellow"
C_ERROR = "red"
C_TEXT = "white"
C_MUTED = "grey58"
C_HEADER = "bold white"
C_BORDER = "grey35"
C_COLNAME = "bold cyan"
C_TYPE = "magenta"
C_PATH = "cyan"

C_ACCENT = C_PRIMARY
C_OK = C_SUCCESS
C_WARN = C_WARNING
C_ERR = C_ERROR
C_HEAD = C_HEADER
C_VAL = C_TEXT
C_ID = C_PRIMARY
C_REF = C_MUTED


def _rule(label: str = "") -> None:
    out.print()
    if label:
        out.print(Text(f"▸ {label.upper()}", style=C_HEADER))
        out.print(Text("─" * out.width, style=C_BORDER))


def _print(renderable: Any) -> None:
    out.print(renderable)


def make_progress() -> tuple[Progress, Any]:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=out,
        transient=False,
    )
    task = progress.add_task("", total=None)
    return progress, task


@contextmanager
def with_spinner(label: str) -> Generator[Progress, None, None]:
    progress, task = make_progress()
    progress.update(task, description=label)
    with progress:
        yield progress


async def run_pipeline(
    stream: AsyncIterator[ProgressEvent],
    on_done: Callable[[ProgressEvent], None] | None = None,
) -> None:
    progress, task = make_progress()

    with progress:
        async for event in stream:
            progress.update(task, description=event.message)

            if event.event_type == EventType.PROGRESS and event.payload:
                if "latency_ms" in event.payload:
                    progress.console.print(
                        Text("✔ ", style=C_SUCCESS)
                        + Text("connected ", style=C_TEXT)
                        + Text(f"{event.payload['latency_ms']:.1f} ms", style=C_MUTED)
                    )
                elif "rows" in event.payload and "columns" in event.payload:
                    progress.console.print(
                        Text("✔ ", style=C_SUCCESS)
                        + Text(f"{event.payload['rows']:,} rows ", style=C_TEXT)
                        + Text(f"{event.payload['columns']} cols", style=C_MUTED)
                    )

            elif event.event_type == EventType.WARNING:
                progress.console.print(
                    Text("⚠ ", style=C_WARNING) + Text(event.message, style=C_WARNING)
                )

            elif event.event_type == EventType.ERROR:
                progress.console.print(
                    Text("✖ ", style=C_ERROR) + Text(event.message, style=C_ERROR)
                )
                raise SystemExit(1)

            elif event.event_type == EventType.DONE:
                if on_done:
                    on_done(event)


def print_connection_result(latency_ms: float | None) -> None:
    if latency_ms is None:
        return
    out.print(
        Text("✔ ", style=C_SUCCESS)
        + Text("connected ", style=C_TEXT)
        + Text(f"{latency_ms:.1f} ms", style=C_MUTED)
    )


def print_read_result(rows: int, columns: int) -> None:
    out.print(
        Text("✔ ", style=C_SUCCESS)
        + Text(f"{rows:,} rows ", style=C_TEXT)
        + Text(f"{columns} cols", style=C_MUTED)
    )


def print_run_summary(payload: dict[str, Any]) -> None:
    _rule("run")
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column(style=C_MUTED, no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_row("run", Text(f"● {payload.get('run_id', '-')}", style=f"bold {C_PRIMARY}"))
    t.add_row("ref", Text(payload.get("ref", "-"), style=C_MUTED))
    out.print(t)
    out.print()


def print_schema(schema: dict[str, str]) -> None:
    if not schema:
        return
    _rule("schema")
    t = Table(box=None, show_header=True, header_style=C_HEADER, padding=(0, 2))
    t.add_column("column", style=C_COLNAME)
    t.add_column("type", justify="right", style=C_TYPE)
    for col, dtype in schema.items():
        t.add_row(f"• {col}", dtype)
    out.print(t)
    out.print()


def print_artifact_path(path: str) -> None:
    _rule("output")
    out.print(
        Text("📦 path ", style=C_MUTED)
        + Text(path, style=f"bold {C_PATH}")
    )
    out.print()


def exit_on_error(event: ProgressEvent) -> None:
    if event.event_type == EventType.ERROR:
        sys.exit(1)