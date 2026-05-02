import sys
from typing import Any

from rich.console import Console
from rich.table import Table, box
from rich.text import Text

from ..core.events import EventType, ProgressEvent

out = Console()
err = Console(stderr=True)


def print_event(event: ProgressEvent) -> None:
    if event.event_type == EventType.STARTED:
        out.print(Text("* ", style="bold blue") + Text(event.message, style="bold"))

    elif event.event_type == EventType.PROGRESS:
        if event.payload and "rows" in event.payload:
            out.print(f"  ~ {event.payload['rows']:,} rows", style="dim")

    elif event.event_type == EventType.WARNING:
        out.print(Text("! ", style="yellow bold") + Text(event.message, style="yellow"))

    elif event.event_type == EventType.ERROR:
        err.print(Text("x ", style="bold red") + Text(event.message, style="red"))


def print_connection_result(latency_ms: float | None) -> None:
    if latency_ms is not None:
        out.print(f"  + connected  [dim]{latency_ms:.1f}ms[/dim]", style="green")


def print_read_result(rows: int, columns: int) -> None:
    out.print()
    out.print(
        Text("  ✓ ", style="bold green")
        + Text(f"{rows:,} rows", style="bold")
        + Text(f"  {columns} cols", style="dim")
    )
    out.print()


def print_run_summary(payload: dict[str, Any]) -> None:
    t = Table(box=None, show_header=False, padding=(0, 2, 0, 0), show_edge=False)
    t.add_column(style="dim", no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_row("run", Text(payload.get("run_id", ""), style="bold"))
    t.add_row("ref", Text(payload.get("ref", ""), style="dim"))
    out.print(t)


def print_schema(schema: dict[str, str]) -> None:
    if not schema:
        return
    out.print()
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold dim", show_edge=False)
    t.add_column("column", style="cyan", no_wrap=True)
    t.add_column("type", style="dim", no_wrap=True)
    for col, dtype in schema.items():
        t.add_row(col, dtype)
    out.print(t)


def print_artifact_path(path: str) -> None:
    out.print(f"  path  [dim]{path}[/dim]")


def exit_on_error(event: ProgressEvent) -> None:
    if event.event_type == EventType.ERROR:
        sys.exit(1)