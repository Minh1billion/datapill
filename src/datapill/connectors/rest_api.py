from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Any
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
    pagination_type: Optional[str] = None
    page_param: str = "page"
    page_size_param: str = "page_size"
    page_size: int = 100
    results_key: Optional[str] = None
    next_key: Optional[str] = "next"
    read_only: bool = False


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

    def _session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(headers=self._headers, auth=self._auth, timeout=self._timeout)

    async def connect(self) -> ConnectionStatus:
        async def probe():
            async with self._session() as session:
                async with session.get(self.config.base_url) as resp:
                    resp.raise_for_status()

        ok, latency_ms, error = await timed_connect(probe)
        return ConnectionStatus(ok=ok, latency_ms=latency_ms, error=error)

    async def cleanup(self) -> None:
        return

    async def query(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        stream: bool = False,
    ) -> pl.DataFrame | AsyncGenerator[pl.DataFrame, Any]:
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        if not stream:
            async with self._session() as session:
                async with session.get(url, params=params or {}) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            rows = convert_to_list_of_dicts(data, self.config.results_key)
            return pl.DataFrame(rows) if rows else pl.DataFrame()

        async def generate() -> AsyncGenerator[pl.DataFrame, Any]:
            async with self._session() as session:
                if self.config.pagination_type == "page":
                    page = 1
                    while True:
                        p = {**(params or {}), self.config.page_param: page, self.config.page_size_param: self.config.page_size}
                        async with session.get(url, params=p) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                        rows = convert_to_list_of_dicts(data, self.config.results_key)
                        if not rows:
                            break
                        yield pl.DataFrame(rows)
                        page += 1

                elif self.config.pagination_type == "cursor":
                    next_url: Optional[str] = url
                    cur_params = params
                    while next_url:
                        async with session.get(next_url, params=cur_params or {}) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                        rows = convert_to_list_of_dicts(data, self.config.results_key)
                        if rows:
                            yield pl.DataFrame(rows)
                        next_url = data.get(self.config.next_key) if isinstance(data, dict) else None
                        cur_params = None

                else:
                    async with session.get(url, params=params or {}) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
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
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        async with self._session() as session:
            async with session.request(method, url, json=payload or {}) as resp:
                resp.raise_for_status()
                return await resp.json()