from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Any
import asyncio
import polars as pl
import aiohttp

from .base import BaseConnector, ConnectionStatus
from ..utils.connection import timed_connect
from ..utils.auth import build_bearer_header, build_basic_auth
from ..utils.normalization import convert_to_list_of_dicts


@dataclass
class RestApiConnectorConfig:
    base_url: str
    headers: dict = field(default_factory=dict)
    auth_type: Optional[str] = None
    auth_token: Optional[str] = None
    basic_user: Optional[str] = None
    basic_password: Optional[str] = None
    timeout_s: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 0.5
    pagination_type: Optional[str] = None
    page_param: str = "page"
    page_size_param: str = "page_size"
    page_size: int = 100
    results_key: Optional[str] = None
    next_key: Optional[str] = "next"
    read_only: bool = False
    concurrent_pages: int = 5
    connector_limit: int = 100


class RestApiConnector(BaseConnector[RestApiConnectorConfig]):
    def __init__(self, config: RestApiConnectorConfig):
        super().__init__(config)
        self._headers = {**config.headers}
        if config.auth_type == "bearer" and config.auth_token:
            self._headers.update(build_bearer_header(config.auth_token))
        self._auth = None
        if config.auth_type == "basic" and config.basic_user and config.basic_password:
            user, password = build_basic_auth(config.basic_user, config.basic_password)
            self._auth = aiohttp.BasicAuth(user, password)
        self._timeout = aiohttp.ClientTimeout(total=config.timeout_s)
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> ConnectionStatus:
        async def probe():
            connector = aiohttp.TCPConnector(limit=self.config.connector_limit)
            self._session = aiohttp.ClientSession(
                headers=self._headers,
                auth=self._auth,
                timeout=self._timeout,
                connector=connector,
            )
            async with self._session.get(self.config.base_url) as resp:
                resp.raise_for_status()

        ok, latency_ms, error = await timed_connect(probe)
        return ConnectionStatus(ok=ok, latency_ms=latency_ms, error=error)

    async def cleanup(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def _get_with_retry(
        self, url: str, params: Optional[dict]
    ) -> Any:
        last_exc: Exception = RuntimeError("no attempts")
        for attempt in range(self.config.max_retries):
            try:
                async with self._session.get(url, params=params or {}) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientResponseError, aiohttp.ServerConnectionError) as exc:
                last_exc = exc
                await asyncio.sleep(self.config.retry_backoff * (2 ** attempt))
        raise last_exc

    async def query(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        stream: bool = False,
    ) -> pl.DataFrame | AsyncGenerator[pl.DataFrame, Any]:
        if not self._session:
            raise ValueError("Not connected")

        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        if not stream:
            data = await self._get_with_retry(url, params)
            rows = convert_to_list_of_dicts(data, self.config.results_key)
            return pl.DataFrame(rows) if rows else pl.DataFrame()

        async def generate() -> AsyncGenerator[pl.DataFrame, Any]:
            if self.config.pagination_type == "page":
                sem = asyncio.Semaphore(self.config.concurrent_pages)
                page = 1
                while True:
                    async def fetch_page(p: int) -> list[dict]:
                        async with sem:
                            d = await self._get_with_retry(
                                url,
                                {
                                    **(params or {}),
                                    self.config.page_param: p,
                                    self.config.page_size_param: self.config.page_size,
                                },
                            )
                            return convert_to_list_of_dicts(d, self.config.results_key)

                    batch_pages = list(range(page, page + self.config.concurrent_pages))
                    results = await asyncio.gather(*[fetch_page(p) for p in batch_pages])
                    empty = False
                    for rows in results:
                        if not rows:
                            empty = True
                            break
                        yield pl.DataFrame(rows)
                    if empty:
                        break
                    page += self.config.concurrent_pages

            elif self.config.pagination_type == "cursor":
                next_url: Optional[str] = url
                cur_params = params
                while next_url:
                    data = await self._get_with_retry(next_url, cur_params)
                    rows = convert_to_list_of_dicts(data, self.config.results_key)
                    if rows:
                        yield pl.DataFrame(rows)
                    next_url = data.get(self.config.next_key) if isinstance(data, dict) else None
                    cur_params = None

            else:
                data = await self._get_with_retry(url, params)
                rows = convert_to_list_of_dicts(data, self.config.results_key)
                if rows:
                    yield pl.DataFrame(rows)

        return generate()

    async def execute(
        self,
        endpoint: str,
        method: str = "POST",
        payload: Optional[dict] = None,
    ) -> dict:
        if self.config.read_only:
            raise PermissionError("read only")
        if not self._session:
            raise ValueError("Not connected")
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        async with self._session.request(method, url, json=payload or {}) as resp:
            resp.raise_for_status()
            return await resp.json()