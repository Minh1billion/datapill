import sys
from contextlib import contextmanager
from typing import Any, AsyncIterator, Callable, Generator, TypeVar, ParamSpec
import functools

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table, box
from rich.text import Text

from ..core.events import EventType, ProgressEvent

out = Console()
err = Console(stderr=True)

C_OK     = "bright_green"
C_WARN   = "yellow"
C_ERR    = "bright_red"
C_VAL    = "bright_white"
C_MUTED  = "grey62"
C_ACCENT = "bright_cyan"
C_KEY    = "bright_yellow"
C_HEAD   = "bold white"
C_RULE   = "grey35"
C_ID     = "bright_magenta"
C_REF    = "cyan"
C_PATH   = "bright_cyan"
C_TYPE   = "bright_blue"


def _rule(label: str = "") -> None:
    out.print()
    if label:
        prefix = f"── {label.upper()} "
        fill = "─" * max(0, out.width - len(prefix))
        out.print(Text(prefix, style=f"bold {C_RULE}") + Text(fill, style=C_RULE))
    else:
        out.print(Rule(style=C_RULE))


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
                        Text("  ✔ connected  ", style=f"bold {C_OK}")
                        + Text(f"{event.payload['latency_ms']:.1f} ms", style=C_MUTED)
                    )
                elif "rows" in event.payload and "columns" in event.payload:
                    progress.console.print(
                        Text("  ✔ ", style=f"bold {C_OK}")
                        + Text(f"{event.payload['rows']:,} rows", style=f"bold {C_VAL}")
                        + Text(f"  {event.payload['columns']} cols", style=C_MUTED)
                    )

            elif event.event_type == EventType.WARNING:
                progress.console.print(
                    Text("⚠ ", style=f"bold {C_WARN}") + Text(event.message, style=C_WARN)
                )

            elif event.event_type == EventType.ERROR:
                progress.console.print(
                    Text("✖ ", style=f"bold {C_ERR}") + Text(event.message, style=C_ERR)
                )
                raise SystemExit(1)

            elif event.event_type == EventType.DONE:
                if on_done:
                    on_done(event)


def print_connection_result(latency_ms: float | None) -> None:
    if latency_ms is None:
        return
    out.print(
        Text("  ✔ connected  ", style=f"bold {C_OK}")
        + Text(f"{latency_ms:.1f} ms", style=C_MUTED)
    )


def print_read_result(rows: int, columns: int) -> None:
    out.print(
        Text("  ✔ ", style=f"bold {C_OK}")
        + Text(f"{rows:,} rows", style=f"bold {C_VAL}")
        + Text(f"  {columns} cols", style=C_MUTED)
    )


def print_run_summary(payload: dict[str, Any]) -> None:
    _rule("run")
    t = Table(box=None, show_header=False, show_edge=False, padding=(0, 1), pad_edge=False)
    t.add_column(style=C_MUTED, no_wrap=True, min_width=6)
    t.add_column(no_wrap=True)
    t.add_row("run", Text(payload.get("run_id", "-"), style=f"bold {C_ID}"))
    t.add_row("ref", Text(payload.get("ref", "-"), style=C_REF))
    out.print(t)


def print_schema(schema: dict[str, str]) -> None:
    if not schema:
        return
    _rule("schema")
    t = Table(
        box=None,
        show_header=True,
        show_lines=False,
        header_style=C_HEAD,
        show_edge=False,
        padding=(0, 1),
        pad_edge=False,
    )
    t.add_column("column", style=f"bold {C_ACCENT}", no_wrap=True)
    t.add_column("type", style=C_TYPE, no_wrap=True)
    for col, dtype in schema.items():
        t.add_row(col, dtype)
    out.print(t)


def print_artifact_path(path: str) -> None:
    _rule("output")
    out.print(Text("  path  ", style=C_MUTED) + Text(path, style=C_PATH))


def exit_on_error(event: ProgressEvent) -> None:
    if event.event_type == EventType.ERROR:
        sys.exit(1)