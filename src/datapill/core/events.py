from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any
import time


class EventType(str, Enum):
    STARTED = "started"
    PROGRESS = "progress"
    DONE = "done"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ProgressEvent:
    event_type: EventType
    message: str
    progress_pct: Optional[float] = None
    eta_seconds: Optional[float] = None
    payload: Optional[dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

    def is_terminal(self) -> bool:
        return self.event_type in (EventType.DONE, EventType.ERROR)