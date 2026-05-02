from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

import polars as pl


class BaseReader(ABC):
    @abstractmethod
    async def read(
        self,
        connector: Any,
        options: dict[str, Any],
        is_sample: bool,
        sample_size: int,
    ) -> AsyncGenerator[pl.DataFrame, None]: ...


class DatabaseReader(BaseReader):
    async def read(
        self,
        connector: Any,
        options: dict[str, Any],
        is_sample: bool,
        sample_size: int,
    ) -> AsyncGenerator[pl.DataFrame, None]:
        query = options["query"]

        if is_sample and "limit" not in query.lower():
            query = f"SELECT * FROM ({query}) AS t LIMIT {sample_size}"

        batch_size = options.get("batch_size")

        result = await connector.query(query, stream=True, batch_size=batch_size)

        if isinstance(result, pl.DataFrame):
            async def _wrap_df():
                yield result
            return _wrap_df()

        return result


class KafkaReader(BaseReader):
    async def read(
        self,
        connector: Any,
        options: dict[str, Any],
        is_sample: bool,
        sample_size: int,
    ) -> AsyncGenerator[pl.DataFrame, None]:
        async def _generate():
            received = 0
            async for df in connector.consume(
                options["topic"],
                max_messages=sample_size if is_sample else None,
                timeout_ms=options.get("timeout_ms", 5000),
            ):
                if is_sample and received >= sample_size:
                    break
                rows_left = sample_size - received if is_sample else len(df)
                yield df.head(rows_left) if is_sample and received + len(df) > sample_size else df
                received += len(df)

        return _generate()


class FileReader(BaseReader):
    async def read(
        self,
        connector: Any,
        options: dict[str, Any],
        is_sample: bool,
        sample_size: int,
    ) -> AsyncGenerator[pl.DataFrame, None]:
        batch_size = options.get("batch_size")

        result = await connector.read(
            options["path"],
            stream=True,
            batch_size=batch_size,
        )

        async def _generate():
            if isinstance(result, pl.DataFrame):
                yield result.head(sample_size) if is_sample else result
                return

            received = 0
            async for df in result:
                if is_sample:
                    remaining = sample_size - received
                    if remaining <= 0:
                        break
                    chunk = df.head(remaining)
                    yield chunk
                    received += len(chunk)
                    if received >= sample_size:
                        break
                else:
                    yield df

        return _generate()


class RestApiReader(BaseReader):
    async def read(
        self,
        connector: Any,
        options: dict[str, Any],
        is_sample: bool,
        sample_size: int,
    ) -> AsyncGenerator[pl.DataFrame, None]:
        endpoint = options.get("endpoint", "")
        params = options.get("params")
        page_size = options.get("page_size", 20)

        result = await connector.query(endpoint, params=params, stream=True)

        async def _generate():
            if isinstance(result, pl.DataFrame):
                yield result.head(sample_size) if is_sample else result
                return

            received = 0
            async for df in result:
                if len(df) == 0:
                    break

                if is_sample:
                    remaining = sample_size - received
                    if remaining <= 0:
                        break
                    chunk = df.head(remaining)
                    yield chunk
                    received += len(chunk)
                    if received >= sample_size:
                        break
                else:
                    yield df
                    received += len(df)

                if len(df) < page_size:
                    break

        return _generate()


_REGISTRY: dict[str, BaseReader] = {
    "postgres": DatabaseReader(),
    "mysql":    DatabaseReader(),
    "sqlite":   DatabaseReader(),
    "kafka":    KafkaReader(),
    "s3":       FileReader(),
    "local":    FileReader(),
    "rest":     RestApiReader(),
}


def get_reader(source: str) -> BaseReader:
    reader = _REGISTRY.get(source)
    if reader is None:
        raise ValueError(f"no reader registered for source: {source!r}")
    return reader