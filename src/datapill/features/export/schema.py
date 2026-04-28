from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
 
 
@dataclass
class WriteConfig:
    format: str
    path: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)
    write_mode: str = "replace"
    primary_keys: list[str] = field(default_factory=list)
    connector_config: dict[str, Any] | None = None
 
 
@dataclass
class ExportResult:
    run_id: str
    rows_written: int
    path: Path | None
    destination: str