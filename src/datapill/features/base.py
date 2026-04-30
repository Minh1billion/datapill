from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Any
from ..core.context import PipelineContext
from ..core.events import ProgressEvent


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    steps: list[dict[str, Any]] = field(default_factory=list)
    estimated_duration_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Pipeline(ABC):

    @abstractmethod
    def validate(self, context: PipelineContext) -> ValidationResult: ...

    @abstractmethod
    def plan(self, context: PipelineContext) -> ExecutionPlan: ...

    @abstractmethod
    async def execute(
        self, plan: ExecutionPlan, context: PipelineContext
    ) -> AsyncGenerator[ProgressEvent, None]: ...

    @abstractmethod
    def serialize(self) -> dict[str, Any]: ...