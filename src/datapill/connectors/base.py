from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Any
import polars as pl


@dataclass
class ColumnMeta:
    name: str
    dtype: str
    nullable: bool


@dataclass
class SchemaInfo:
    columns: list[ColumnMeta]
    row_count_estimate: int | None = None


@dataclass
class ConnectionStatus:
    ok: bool
    latency_ms: float | None = None
    error: str | None = None


@dataclass
class WriteResult:
    rows_written: int
    duration_s: float


class BaseConnector(ABC):

    @abstractmethod
    async def read(self, query: dict[str, Any], options: dict[str, Any] | None = None) -> pl.DataFrame: ...

    @abstractmethod
    async def read_stream(
        self, query: dict[str, Any], options: dict[str, Any] | None = None
    ) -> AsyncGenerator[pl.DataFrame, None]: ...

    @abstractmethod
    async def schema(self) -> SchemaInfo: ...

    @abstractmethod
    async def test_connection(self) -> ConnectionStatus: ...

    @abstractmethod
    async def write(
        self, df: pl.DataFrame, target: dict[str, Any], options: dict[str, Any] | None = None
    ) -> WriteResult: ...

    async def close(self) -> None:
        pass