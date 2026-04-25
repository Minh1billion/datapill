import json
import pytest
import polars as pl
from pytest_httpserver import HTTPServer

from dataprep.connectors.rest import RESTConnector


def _make_config(server: HTTPServer, **extra) -> dict:
    return {
        "base_url": server.url_for(""),
        "headers": {"Content-Type": "application/json"},
        "timeout_seconds": 5,
        "max_retries": 2,
        **extra,
    }


def _make_sample_rows(n: int = 10) -> list[dict]:
    return [{"id": i, "name": f"item_{i}", "value": i * 1.5} for i in range(1, n + 1)]


SAMPLE_ROWS = _make_sample_rows(10)
_SAMPLE_COLS = set(SAMPLE_ROWS[0].keys())


@pytest.fixture
def simple_server(httpserver: HTTPServer):
    httpserver.expect_request("/items").respond_with_json(SAMPLE_ROWS)
    return httpserver


@pytest.fixture
def nested_server(httpserver: HTTPServer):
    httpserver.expect_request("/data").respond_with_json({"results": SAMPLE_ROWS})
    return httpserver


@pytest.fixture
def paginated_offset_server(httpserver: HTTPServer):
    page_size = len(SAMPLE_ROWS) // 2
    page1 = SAMPLE_ROWS[:page_size]
    page2 = SAMPLE_ROWS[page_size:]
    httpserver.expect_request("/items", query_string=f"limit={page_size}&offset=0").respond_with_json(page1)
    httpserver.expect_request("/items", query_string=f"limit={page_size}&offset={page_size}").respond_with_json(page2)
    httpserver.expect_request("/items", query_string=f"limit={page_size}&offset={len(SAMPLE_ROWS)}").respond_with_json([])
    return httpserver, page_size


@pytest.fixture
def health_server(httpserver: HTTPServer):
    httpserver.expect_request("/health").respond_with_data("ok", status=200)
    return httpserver


@pytest.mark.asyncio
async def test_read_flat_list(simple_server: HTTPServer):
    connector = RESTConnector(_make_config(simple_server))
    df = await connector.read({"endpoint": "/items"})
    assert len(df) == len(SAMPLE_ROWS)
    assert _SAMPLE_COLS.issubset(set(df.columns))


@pytest.mark.asyncio
async def test_read_nested_response_path(nested_server: HTTPServer):
    connector = RESTConnector(_make_config(nested_server, response_path="results"))
    df = await connector.read({"endpoint": "/data"})
    assert len(df) == len(SAMPLE_ROWS)


@pytest.mark.asyncio
async def test_read_offset_pagination(paginated_offset_server):
    httpserver, page_size = paginated_offset_server
    connector = RESTConnector(
        _make_config(
            httpserver,
            pagination={"type": "offset", "limit": page_size, "limit_param": "limit", "offset_param": "offset"},
        )
    )
    df = await connector.read({"endpoint": "/items"})
    assert len(df) == len(SAMPLE_ROWS)


@pytest.mark.asyncio
async def test_read_stream_yields_per_page(paginated_offset_server):
    httpserver, page_size = paginated_offset_server
    connector = RESTConnector(
        _make_config(
            httpserver,
            pagination={"type": "offset", "limit": page_size, "limit_param": "limit", "offset_param": "offset"},
        )
    )
    frames = []
    async for chunk in connector.read_stream({"endpoint": "/items"}):
        frames.append(chunk)
    assert len(frames) == 2
    assert sum(len(f) for f in frames) == len(SAMPLE_ROWS)


@pytest.mark.asyncio
async def test_schema_from_endpoint(nested_server: HTTPServer):
    connector = RESTConnector(
        _make_config(nested_server, response_path="results", default_endpoint="/data")
    )
    info = await connector.schema()
    names = set(c.name for c in info.columns)
    assert _SAMPLE_COLS.issubset(names)


@pytest.mark.asyncio
async def test_schema_no_endpoint():
    connector = RESTConnector({"base_url": "http://localhost", "timeout_seconds": 1})
    info = await connector.schema()
    assert info.columns == []


@pytest.mark.asyncio
async def test_test_connection_healthy(health_server: HTTPServer):
    connector = RESTConnector(
        _make_config(health_server, health_endpoint="/health")
    )
    status = await connector.test_connection()
    assert status.ok
    assert status.latency_ms >= 0


@pytest.mark.asyncio
async def test_test_connection_failure():
    connector = RESTConnector(
        {"base_url": "http://localhost:19999", "health_endpoint": "/", "timeout_seconds": 1, "max_retries": 1}
    )
    status = await connector.test_connection()
    assert not status.ok
    assert status.error is not None


@pytest.mark.asyncio
async def test_write_posts_each_row(httpserver: HTTPServer, sample_df: pl.DataFrame):
    posted = []

    def handler(request):
        posted.append(request.get_json())
        from werkzeug.wrappers import Response
        return Response(status=200)

    httpserver.expect_request("/records", method="POST").respond_with_handler(handler)
    connector = RESTConnector(_make_config(httpserver))
    n = min(3, len(sample_df))
    small = sample_df.head(n)
    result = await connector.write(small, {"endpoint": "/records"})
    assert result.rows_written == n
    assert len(posted) == n


@pytest.mark.asyncio
async def test_write_returns_duration(httpserver: HTTPServer, sample_df: pl.DataFrame):
    httpserver.expect_request("/records", method="POST").respond_with_json({})
    connector = RESTConnector(_make_config(httpserver))
    result = await connector.write(sample_df.head(1), {"endpoint": "/records"})
    assert result.duration_s >= 0


@pytest.mark.asyncio
async def test_rate_limit_retry(httpserver: HTTPServer):
    call_count = {"n": 0}
    retry_rows = SAMPLE_ROWS[:2]

    def handler(request):
        from werkzeug.wrappers import Response
        call_count["n"] += 1
        if call_count["n"] < 2:
            return Response(status=429, headers={"Retry-After": "0"})
        return Response(
            response=json.dumps(retry_rows),
            status=200,
            content_type="application/json",
        )

    httpserver.expect_request("/retry").respond_with_handler(handler)
    connector = RESTConnector(_make_config(httpserver, max_retries=3))
    df = await connector.read({"endpoint": "/retry"})
    assert len(df) == len(retry_rows)
    assert call_count["n"] >= 2