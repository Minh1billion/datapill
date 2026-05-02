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

BOLD_WHITE = "bold white"
ORANGE = "bold #FF9900"
GRAY = "bright_black"
WHITE = "white"
GREEN = "green"
RED = "red"
YELLOW = "yellow"
CYAN = "cyan"
MAGENTA = "magenta"
DARK_BLUE = "bold #232F3E"


def _rule(label: str = "") -> None:
    out.print()
    if label:
        out.print(Text(f"▸ {label.upper()}", style=BOLD_WHITE))
        out.print(Text("─" * out.width, style=GRAY))


def _print(renderable: Any) -> None:
    out.print(renderable)


def make_progress() -> tuple[Progress, Any]:
    progress = Progress(
        SpinnerColumn(spinner_name="dots", style=ORANGE),
        TextColumn("[progress.description]{task.description}", style=WHITE),
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
                        Text("[OK] ", style=GREEN)
                        + Text("connected ", style=WHITE)
                        + Text(f"{event.payload['latency_ms']:.1f} ms", style=GRAY)
                    )
                elif "rows" in event.payload and "columns" in event.payload:
                    progress.console.print(
                        Text("[OK] ", style=GREEN)
                        + Text(f"{event.payload['rows']:,} rows ", style=WHITE)
                        + Text(f"{event.payload['columns']} cols", style=GRAY)
                    )
            elif event.event_type == EventType.WARNING:
                progress.console.print(
                    Text("[WARN] ", style=YELLOW) + Text(event.message, style=YELLOW)
                )
            elif event.event_type == EventType.ERROR:
                progress.console.print(
                    Text("[FAIL] ", style=RED) + Text(event.message, style=RED)
                )
                raise SystemExit(1)
            elif event.event_type == EventType.DONE:
                if on_done:
                    on_done(event)


def print_connection_result(latency_ms: float | None) -> None:
    if latency_ms is None:
        return
    out.print(
        Text("[OK] ", style=GREEN)
        + Text("connected ", style=WHITE)
        + Text(f"{latency_ms:.1f} ms", style=GRAY)
    )


def print_read_result(rows: int, columns: int) -> None:
    out.print(
        Text("[OK] ", style=GREEN)
        + Text(f"{rows:,} rows ", style=WHITE)
        + Text(f"{columns} cols", style=GRAY)
    )


def print_run_summary(payload: dict[str, Any]) -> None:
    _rule("run")
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column(style=GRAY, no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_row("run:", Text(f"{payload.get('run_id', '-')}", style=f"bold {ORANGE}"))
    t.add_row("ref:", Text(payload.get("ref", "-"), style=f"bold {ORANGE}"))
    out.print(t)
    out.print()


def print_schema(schema: dict[str, str]) -> None:
    if not schema:
        return
    _rule("schema")
    t = Table(box=None, show_header=True, header_style=BOLD_WHITE, padding=(0, 2))
    t.add_column("column", style=CYAN)
    t.add_column("type", justify="right", style=GRAY)
    for col, dtype in schema.items():
        t.add_row(f"• {col}", dtype)
    out.print(t)
    out.print()


def print_artifact_path(path: str) -> None:
    _rule("output")
    out.print(
        Text("📦 path: ", style=GRAY)
        + Text(path, style=f"bold {ORANGE}")
    )
    out.print()


def exit_on_error(event: ProgressEvent) -> None:
    if event.event_type == EventType.ERROR:
        sys.exit(1)