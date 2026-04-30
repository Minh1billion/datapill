from dataclasses import dataclass, field
from typing import Any
import uuid

from ..storage.artifact_store import ArtifactStore

@dataclass
class Context:
    artifact_store: ArtifactStore
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)