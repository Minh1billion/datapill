from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class BaseReader(ABC):
    @abstractmethod
    async def read(
        self,
        connector: Any,
        options: dict[str, Any],
        is_sample: bool,
        sample_size: int,
    ) -> pl.DataFrame: ...


class DatabaseReader(BaseReader):
    async def read(self, connector, options, is_sample, sample_size) -> pl.DataFrame:
        table = options.get("table")
        query = options.get("query") or (
            f"SELECT * FROM {table} LIMIT {sample_size}"
            if is_sample
            else f"SELECT * FROM {table}"
        )
        result = await connector.query(query)
        if isinstance(result, pl.DataFrame):
            return result
        return pl.concat([df async for df in result])


class KafkaReader(BaseReader):
    async def read(self, connector, options, is_sample, sample_size) -> pl.DataFrame:
        batches = []
        async for df in connector.consume(
            options["topic"],
            max_messages=sample_size if is_sample else None,
        ):
            batches.append(df)
        return pl.concat(batches) if batches else pl.DataFrame()


class FileReader(BaseReader):
    async def read(self, connector, options, is_sample, sample_size) -> pl.DataFrame:
        result = await connector.read(options["path"])
        if isinstance(result, pl.DataFrame):
            return result.head(sample_size) if is_sample else result
        batches, count = [], 0
        async for df in result:
            batches.append(df)
            count += len(df)
            if is_sample and count >= sample_size:
                break
        full = pl.concat(batches) if batches else pl.DataFrame()
        return full.head(sample_size) if is_sample else full


_REGISTRY: dict[str, BaseReader] = {
    "postgres":  DatabaseReader(),
    "mysql":     DatabaseReader(),
    "sqlite":    DatabaseReader(),
    "kafka":     KafkaReader(),
    "s3":        FileReader(),
    "local":     FileReader(),
}


def get_reader(source: str) -> BaseReader:
    reader = _REGISTRY.get(source)
    if reader is None:
        raise ValueError(f"no reader registered for source: {source!r}")
    return reader