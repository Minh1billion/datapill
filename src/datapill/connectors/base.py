from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass
class ConnectionStatus:
    ok: bool
    latency_ms: float | None = None
    error: str | None = None

class BaseConnector(ABC, Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config

    @abstractmethod
    async def connect(self) -> ConnectionStatus:
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        ...