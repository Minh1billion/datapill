import asyncio
import datetime
import time
from typing import AsyncGenerator, Any

import aiohttp
import polars as pl

from .base import BaseConnector, ColumnMeta, ConnectionStatus, SchemaInfo, WriteResult


def _extract_by_path(data: Any, path: str) -> list[dict]:
    if not path:
        if isinstance(data, list):
            return data
        return [data]
    for key in path.split("."):
        if isinstance(data, dict):
            data = data.get(key, [])
        else:
            return []
    if isinstance(data, list):
        return data
    return [data]


def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _serialize_record(record: dict) -> dict:
    out = {}
    for k, v in record.items():
        if isinstance(v, (datetime.date, datetime.datetime)):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


class RESTConnector(BaseConnector):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._base_url = config["base_url"].rstrip("/")
        self._headers = config.get("headers", {})
        self._pagination = config.get("pagination", {})
        self._response_path = config.get("response_path", "")
        self._rate_limit_delay = config.get("rate_limit_delay", 0.0)
        self._timeout = aiohttp.ClientTimeout(total=config.get("timeout_seconds", 30))

    def _session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(headers=self._headers, timeout=self._timeout)

    async def _request_with_retry(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        **kwargs,
    ) -> Any:
        max_retries = self.config.get("max_retries", 3)
        base_backoff = 0.5
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                async with session.request(method, url, **kwargs) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        retry_after = float(resp.headers.get("Retry-After", base_backoff * (2**attempt)))
                        await asyncio.sleep(retry_after)
                        last_exc = aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                        continue
                    resp.raise_for_status()
                    if self._rate_limit_delay:
                        await asyncio.sleep(self._rate_limit_delay)
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        return await resp.json()
                    return await resp.text()
            except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError) as exc:
                last_exc = exc
                await asyncio.sleep(base_backoff * (2**attempt))

        raise last_exc

    async def _collect_pages(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        params: dict,
    ) -> list[list[dict]]:
        url = f"{self._base_url}{endpoint}"
        pagination_type = self._pagination.get("type", "none")
        pages: list[list[dict]] = []

        if pagination_type == "offset":
            limit = self._pagination.get("limit", 100)
            limit_param = self._pagination.get("limit_param", "limit")
            offset_param = self._pagination.get("offset_param", "offset")
            offset = 0
            while True:
                p = {**params, limit_param: limit, offset_param: offset}
                data = await self._request_with_retry(session, "GET", url, params=p)
                rows = _extract_by_path(data, self._response_path)
                if not rows:
                    break
                pages.append(rows)
                if len(rows) < limit:
                    break
                offset += limit

        elif pagination_type == "cursor":
            cursor_path = self._pagination.get("cursor_path", "next_cursor")
            cursor_param = self._pagination.get("cursor_param", "cursor")
            cursor = None
            while True:
                p = {**params}
                if cursor:
                    p[cursor_param] = cursor
                data = await self._request_with_retry(session, "GET", url, params=p)
                rows = _extract_by_path(data, self._response_path)
                if not rows:
                    break
                pages.append(rows)
                cursor = data.get(cursor_path)
                if not cursor:
                    break

        elif pagination_type == "link_header":
            current_url = url
            while current_url:
                async with session.get(current_url, params=params if current_url == url else {}) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    rows = _extract_by_path(data, self._response_path)
                    if rows:
                        pages.append(rows)
                    link = resp.headers.get("Link", "")
                    current_url = _parse_link_next(link)

        else:
            data = await self._request_with_retry(session, "GET", url, params=params)
            rows = _extract_by_path(data, self._response_path)
            if rows:
                pages.append(rows)

        return pages

    async def read(
        self, query: dict[str, Any], options: dict[str, Any] | None = None
    ) -> pl.DataFrame:
        endpoint = query.get("endpoint", "")
        params = query.get("params", {})
        async with self._session() as session:
            pages = await self._collect_pages(session, endpoint, params)
        if not pages:
            return pl.DataFrame()
        frames = [pl.DataFrame(rows) for rows in pages]
        return pl.concat(frames)

    async def read_stream(
        self, query: dict[str, Any], options: dict[str, Any] | None = None
    ) -> AsyncGenerator[pl.DataFrame, None]:
        endpoint = query.get("endpoint", "")
        params = query.get("params", {})
        async with self._session() as session:
            pages = await self._collect_pages(session, endpoint, params)
        for rows in pages:
            yield pl.DataFrame(rows)

    async def schema(self) -> SchemaInfo:
        endpoint = self.config.get("default_endpoint", "")
        if not endpoint:
            return SchemaInfo(columns=[])
        async with self._session() as session:
            data = await self._request_with_retry(session, "GET", f"{self._base_url}{endpoint}")
        rows = _extract_by_path(data, self._response_path)
        if not rows:
            return SchemaInfo(columns=[])
        df = pl.DataFrame(rows[:1])
        columns = [
            ColumnMeta(name=col, dtype=str(dtype), nullable=True)
            for col, dtype in zip(df.columns, df.dtypes)
        ]
        return SchemaInfo(columns=columns)

    async def test_connection(self) -> ConnectionStatus:
        t0 = time.perf_counter()
        try:
            health = self.config.get("health_endpoint", "/")
            async with self._session() as session:
                async with session.get(f"{self._base_url}{health}") as resp:
                    resp.raise_for_status()
            return ConnectionStatus(ok=True, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return ConnectionStatus(ok=False, error=str(exc))

    async def write(
        self,
        df: pl.DataFrame,
        target: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> WriteResult:
        opts = options or {}
        endpoint = target["endpoint"]
        method = opts.get("method", "POST")
        url = f"{self._base_url}{endpoint}"
        records = [_serialize_record(r) for r in df.to_dicts()]
        t0 = time.perf_counter()
        async with self._session() as session:
            for record in records:
                await self._request_with_retry(session, method, url, json=record)
        return WriteResult(rows_written=len(records), duration_s=time.perf_counter() - t0)

    async def close(self) -> None:
        pass


def _parse_link_next(link_header: str) -> str | None:
    for part in link_header.split(","):
        parts = [p.strip() for p in part.split(";")]
        if len(parts) == 2 and parts[1] == 'rel="next"':
            return parts[0].strip("<>")
    return None