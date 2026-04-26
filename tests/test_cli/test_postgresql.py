import pytest

from conftest import assert_connector_ok, assert_ingest_ok, dp, skip_if_down


@pytest.fixture(autouse=True)
def require_postgres():
    skip_if_down("localhost", 5433)


def test_connector_test(pg_config):
    result = dp("connector", "test", "--source", "postgresql", "--config", str(pg_config))
    assert_connector_ok(result)


def test_connector_schema(pg_config):
    result = dp("connector", "schema", "--source", "postgresql", "--config", str(pg_config), "--table", "test_table")
    assert_connector_ok(result)


def test_ingest_table(pg_query_config, out_dir):
    result = dp("ingest", "--source", "postgresql", "--config", str(pg_query_config), out_dir=out_dir)
    assert_ingest_ok(result)