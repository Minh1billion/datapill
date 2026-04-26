import pytest

from conftest import assert_connector_ok, dp, skip_if_down


@pytest.fixture(autouse=True)
def require_mysql():
    skip_if_down("localhost", 3307)


def test_connector_test(my_config):
    result = dp("connector", "test", "--source", "mysql", "--config", str(my_config))
    assert_connector_ok(result)