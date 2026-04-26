from .conftest import assert_connector_ok, assert_ingest_ok, dp


def test_connector_test(rest_config):
    result = dp("connector", "test", "--source", "rest", "--config", str(rest_config))
    assert_connector_ok(result)


def test_connector_schema(rest_config):
    result = dp("connector", "schema", "--source", "rest", "--config", str(rest_config), "--endpoint", "todos")
    assert_connector_ok(result)


def test_ingest_todos(rest_config, out_dir):
    result = dp("ingest", "--source", "rest", "--config", str(rest_config), "--endpoint", "todos", "--batch-size", "10", out_dir=out_dir)
    assert_ingest_ok(result)