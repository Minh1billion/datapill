from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import polars as pl
import pytest

from dataprep.connectors.rest import RESTConnector, _extract_by_path, _parse_link_next

BASE_URL = "https://api.example.com"

SAMPLE_ROWS = [
    {"id": 1, "name": "Alice", "score": 9.5},
    {"id": 2, "name": "Bob", "score": 8.0},
    {"id": 3, "name": "Carol", "score": 7.5},
]


def _make_connector(extra: dict | None = None) -> RESTConnector:
    return RESTConnector({"base_url": BASE_URL, **(extra or {})})


def _make_resp_cm(body, *, status: int = 200, headers: dict | None = None):
    """
    RESTConnector dùng:
        async with session.request(method, url, **kwargs) as resp:
        async with session.get(url) as resp:

    Nên session.request() / session.get() phải trả về một context manager
    (object có __aenter__ / __aexit__), không phải coroutine.
    Hàm này tạo đúng loại đó.
    """
    resp = MagicMock()
    resp.status = status
    resp.headers = headers or {}
    resp.json = AsyncMock(return_value=body)
    resp.raise_for_status = MagicMock()
    resp.request_info = MagicMock()
    resp.history = []
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_session(responses: list) -> tuple[MagicMock, MagicMock]:
    """
    Trả về (session_factory, session).
    session_factory là callable: session_factory() → ctx (async context manager).
    ctx.__aenter__ → session.
    session.request(method, url) → resp_cm (sync call, trả về context manager).
    session.get(url)             → resp_cm (sync call, trả về context manager).
    """
    it = iter(responses)

    def _request(method, url, **kwargs):
        try:
            payload = next(it)
        except StopIteration:
            payload = responses[-1]
        return _make_resp_cm(
            payload.get("body"),
            status=payload.get("status", 200),
            headers=payload.get("headers", {}),
        )

    session = MagicMock()
    session.request = _request
    session.get = _request

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)

    return MagicMock(return_value=ctx), session


def _make_custom_session(request_fn) -> MagicMock:
    """Helper cho test cần custom request function."""
    session = MagicMock()
    session.request = request_fn
    session.get = request_fn
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


class TestRESTConnectorRead:
    @pytest.mark.asyncio
    async def test_read_returns_dataframe(self):
        connector = _make_connector({"response_path": "data"})
        factory, _ = _make_session([{"body": {"data": SAMPLE_ROWS}}])
        with patch.object(connector, "_session", factory):
            result = await connector.read({"endpoint": "/users"})
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_read_columns_match_response(self):
        connector = _make_connector({"response_path": "data"})
        factory, _ = _make_session([{"body": {"data": SAMPLE_ROWS}}])
        with patch.object(connector, "_session", factory):
            result = await connector.read({"endpoint": "/users"})
        assert set(result.columns) == {"id", "name", "score"}

    @pytest.mark.asyncio
    async def test_read_empty_response_returns_empty_dataframe(self):
        connector = _make_connector({"response_path": "data"})
        factory, _ = _make_session([{"body": {"data": []}}])
        with patch.object(connector, "_session", factory):
            result = await connector.read({"endpoint": "/users"})
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_read_no_response_path_uses_root_list(self):
        connector = _make_connector()
        factory, _ = _make_session([{"body": SAMPLE_ROWS}])
        with patch.object(connector, "_session", factory):
            result = await connector.read({"endpoint": "/users"})
        assert len(result) == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_read_passes_params_to_request(self):
        connector = _make_connector()
        captured_params: dict = {}

        def _request(method, url, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return _make_resp_cm(SAMPLE_ROWS)

        factory = _make_custom_session(_request)
        with patch.object(connector, "_session", factory):
            await connector.read({"endpoint": "/users", "params": {"active": True}})

        assert captured_params.get("active") is True


class TestRESTConnectorPagination:
    @pytest.mark.asyncio
    async def test_offset_pagination_concatenates_pages(self):
        connector = _make_connector({
            "pagination": {"type": "offset", "limit": 2, "limit_param": "limit", "offset_param": "offset"},
        })
        factory, _ = _make_session([
            {"body": SAMPLE_ROWS[:2]},
            {"body": SAMPLE_ROWS[2:]},
            {"body": []},
        ])
        with patch.object(connector, "_session", factory):
            result = await connector.read({"endpoint": "/items"})
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_offset_pagination_stops_when_fewer_rows_than_limit(self):
        connector = _make_connector({
            "pagination": {"type": "offset", "limit": 10},
        })
        factory, _ = _make_session([{"body": SAMPLE_ROWS}])
        with patch.object(connector, "_session", factory):
            result = await connector.read({"endpoint": "/items"})
        assert len(result) == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_cursor_pagination_follows_cursor(self):
        connector = _make_connector({
            "pagination": {"type": "cursor", "cursor_path": "next_cursor", "cursor_param": "cursor"},
            "response_path": "items",
        })
        factory, _ = _make_session([
            {"body": {"items": SAMPLE_ROWS[:2], "next_cursor": "tok_abc"}},
            {"body": {"items": SAMPLE_ROWS[2:], "next_cursor": None}},
        ])
        with patch.object(connector, "_session", factory):
            result = await connector.read({"endpoint": "/items"})
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_cursor_pagination_stops_on_missing_cursor(self):
        connector = _make_connector({
            "pagination": {"type": "cursor", "cursor_path": "next"},
            "response_path": "data",
        })
        factory, _ = _make_session([{"body": {"data": SAMPLE_ROWS}}])
        with patch.object(connector, "_session", factory):
            result = await connector.read({"endpoint": "/items"})
        assert len(result) == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_no_pagination_returns_single_page(self):
        connector = _make_connector({"pagination": {"type": "none"}})
        factory, _ = _make_session([{"body": SAMPLE_ROWS}])
        with patch.object(connector, "_session", factory):
            result = await connector.read({"endpoint": "/items"})
        assert len(result) == len(SAMPLE_ROWS)


class TestRESTConnectorStream:
    @pytest.mark.asyncio
    async def test_stream_yields_dataframe_chunks(self):
        connector = _make_connector({
            "pagination": {"type": "offset", "limit": 2},
        })
        factory, _ = _make_session([
            {"body": SAMPLE_ROWS[:2]},
            {"body": SAMPLE_ROWS[2:]},
            {"body": []},
        ])
        chunks = []
        with patch.object(connector, "_session", factory):
            async for chunk in connector.read_stream({"endpoint": "/items"}):
                chunks.append(chunk)
        assert len(chunks) == 2
        assert all(isinstance(c, pl.DataFrame) for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_total_rows_match(self):
        connector = _make_connector({
            "pagination": {"type": "offset", "limit": 2},
        })
        factory, _ = _make_session([
            {"body": SAMPLE_ROWS[:2]},
            {"body": SAMPLE_ROWS[2:]},
            {"body": []},
        ])
        total = 0
        with patch.object(connector, "_session", factory):
            async for chunk in connector.read_stream({"endpoint": "/items"}):
                total += len(chunk)
        assert total == len(SAMPLE_ROWS)


class TestRESTConnectorRetry:
    @pytest.mark.asyncio
    async def test_retries_on_500_then_succeeds(self):
        connector = _make_connector({"max_retries": 3})
        call_count = 0

        def _request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                resp = _make_resp_cm(None, status=500, headers={"Retry-After": "0"})
                resp.raise_for_status = MagicMock(
                    side_effect=aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=500
                    )
                )
                return resp
            return _make_resp_cm(SAMPLE_ROWS)

        factory = _make_custom_session(_request)
        with patch.object(connector, "_session", factory):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await connector.read({"endpoint": "/items"})

        assert call_count == 3
        assert len(result) == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exhausted(self):
        connector = _make_connector({"max_retries": 2})
        call_count = 0

        def _request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_resp_cm(None, status=503, headers={"Retry-After": "0"})

        factory = _make_custom_session(_request)
        with patch.object(connector, "_session", factory):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(Exception):
                    await connector.read({"endpoint": "/items"})

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        connector = _make_connector({"max_retries": 3})
        call_count = 0

        def _request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise aiohttp.ClientConnectionError("refused")
            return _make_resp_cm(SAMPLE_ROWS)

        factory = _make_custom_session(_request)
        with patch.object(connector, "_session", factory):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await connector.read({"endpoint": "/items"})

        assert call_count == 3
        assert len(result) == len(SAMPLE_ROWS)


class TestRESTConnectorWrite:
    @pytest.mark.asyncio
    async def test_write_returns_correct_row_count(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS)
        factory, _ = _make_session([{"body": {}} for _ in SAMPLE_ROWS])
        with patch.object(connector, "_session", factory):
            result = await connector.write(df, {"endpoint": "/users"})
        assert result.rows_written == len(SAMPLE_ROWS)

    @pytest.mark.asyncio
    async def test_write_duration_is_non_negative(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS)
        factory, _ = _make_session([{"body": {}} for _ in SAMPLE_ROWS])
        with patch.object(connector, "_session", factory):
            result = await connector.write(df, {"endpoint": "/users"})
        assert result.duration_s >= 0

    @pytest.mark.asyncio
    async def test_write_uses_post_by_default(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS[:1])
        methods_used: list[str] = []

        def _request(method, url, **kwargs):
            methods_used.append(method)
            return _make_resp_cm({})

        factory = _make_custom_session(_request)
        with patch.object(connector, "_session", factory):
            await connector.write(df, {"endpoint": "/users"})

        assert methods_used == ["POST"]

    @pytest.mark.asyncio
    async def test_write_respects_method_option(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS[:1])
        methods_used: list[str] = []

        def _request(method, url, **kwargs):
            methods_used.append(method)
            return _make_resp_cm({})

        factory = _make_custom_session(_request)
        with patch.object(connector, "_session", factory):
            await connector.write(df, {"endpoint": "/users"}, options={"method": "PUT"})

        assert methods_used == ["PUT"]

    @pytest.mark.asyncio
    async def test_write_sends_one_request_per_row(self):
        connector = _make_connector()
        df = pl.DataFrame(SAMPLE_ROWS)
        call_count = 0

        def _request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_resp_cm({})

        factory = _make_custom_session(_request)
        with patch.object(connector, "_session", factory):
            await connector.write(df, {"endpoint": "/users"})

        assert call_count == len(SAMPLE_ROWS)


class TestRESTConnectorSchema:
    @pytest.mark.asyncio
    async def test_schema_returns_columns_from_default_endpoint(self):
        connector = _make_connector({
            "default_endpoint": "/users",
            "response_path": "data",
        })
        factory, _ = _make_session([{"body": {"data": SAMPLE_ROWS}}])
        with patch.object(connector, "_session", factory):
            schema = await connector.schema()
        assert len(schema.columns) == 3
        assert {c.name for c in schema.columns} == {"id", "name", "score"}

    @pytest.mark.asyncio
    async def test_schema_without_default_endpoint_returns_empty(self):
        connector = _make_connector()
        schema = await connector.schema()
        assert schema.columns == []

    @pytest.mark.asyncio
    async def test_schema_columns_have_nullable_true(self):
        connector = _make_connector({
            "default_endpoint": "/users",
            "response_path": "data",
        })
        factory, _ = _make_session([{"body": {"data": SAMPLE_ROWS}}])
        with patch.object(connector, "_session", factory):
            schema = await connector.schema()
        assert all(c.nullable for c in schema.columns)

    @pytest.mark.asyncio
    async def test_schema_empty_response_returns_empty_columns(self):
        connector = _make_connector({
            "default_endpoint": "/users",
            "response_path": "data",
        })
        factory, _ = _make_session([{"body": {"data": []}}])
        with patch.object(connector, "_session", factory):
            schema = await connector.schema()
        assert schema.columns == []


class TestRESTConnectorConnection:
    @pytest.mark.asyncio
    async def test_healthy_endpoint_returns_ok(self):
        connector = _make_connector({"health_endpoint": "/health"})

        def _get(url, **kwargs):
            return _make_resp_cm({})

        factory = _make_custom_session(_get)
        with patch.object(connector, "_session", factory):
            status = await connector.test_connection()

        assert status.ok is True
        assert status.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_failed_endpoint_returns_not_ok(self):
        connector = _make_connector({"health_endpoint": "/health"})

        def _get(url, **kwargs):
            raise aiohttp.ClientConnectionError("refused")

        factory = _make_custom_session(_get)
        with patch.object(connector, "_session", factory):
            status = await connector.test_connection()

        assert status.ok is False
        assert status.error is not None

    @pytest.mark.asyncio
    async def test_default_health_endpoint_hits_root(self):
        connector = _make_connector()
        visited: list[str] = []

        def _get(url, **kwargs):
            visited.append(url)
            return _make_resp_cm({})

        factory = _make_custom_session(_get)
        with patch.object(connector, "_session", factory):
            await connector.test_connection()

        assert len(visited) == 1
        assert visited[0] == f"{BASE_URL}/"


class TestExtractByPath:
    def test_empty_path_list_returned_as_is(self):
        data = [{"a": 1}, {"a": 2}]
        assert _extract_by_path(data, "") == data

    def test_single_key_extracts_nested_list(self):
        data = {"results": [{"id": 1}, {"id": 2}]}
        assert _extract_by_path(data, "results") == [{"id": 1}, {"id": 2}]

    def test_dotted_path_traverses_nested_dicts(self):
        data = {"meta": {"data": [{"x": 1}]}}
        assert _extract_by_path(data, "meta.data") == [{"x": 1}]

    def test_missing_key_returns_empty_list(self):
        data = {"other": [1, 2]}
        assert _extract_by_path(data, "missing") == []

    def test_non_list_leaf_wrapped_in_list(self):
        data = {"item": {"id": 1}}
        assert _extract_by_path(data, "item") == [{"id": 1}]


class TestParseLinkNext:
    def test_parses_next_rel(self):
        header = '<https://api.example.com/items?page=2>; rel="next"'
        assert _parse_link_next(header) == "https://api.example.com/items?page=2"

    def test_returns_none_when_no_next(self):
        header = '<https://api.example.com/items?page=1>; rel="prev"'
        assert _parse_link_next(header) is None

    def test_returns_none_for_empty_header(self):
        assert _parse_link_next("") is None

    def test_picks_next_from_multiple_rels(self):
        header = (
            '<https://api.example.com/items?page=1>; rel="prev", '
            '<https://api.example.com/items?page=3>; rel="next"'
        )
        assert _parse_link_next(header) == "https://api.example.com/items?page=3"