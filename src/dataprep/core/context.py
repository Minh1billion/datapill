from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import uuid

if TYPE_CHECKING:
    from dataprep.storage.artifact import ArtifactStore


@dataclass
class PipelineContext:
    artifact_store: "ArtifactStore"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)