from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepConfig:
    step: str
    columns: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    step: str
    columns_affected: list[str]
    row_delta: int
    null_delta: dict[str, int]
    dtype_changes: dict[str, tuple[str, str]]
    error: str | None = None


@dataclass
class RunReport:
    run_id: str
    dry_run: bool
    steps: list[StepResult]
    preview_rows: list[dict] | None = None
    output_schema: dict[str, str] | None = None
    warnings: list[str] = field(default_factory=list)