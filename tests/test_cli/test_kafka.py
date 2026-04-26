import pytest

from conftest import assert_connector_ok, dp, skip_if_down


@pytest.fixture(autouse=True)
def require_kafka():
    skip_if_down("localhost", 9092)


def test_connector_test(kafka_config):
    result = dp("connector", "test", "--source", "kafka", "--config", str(kafka_config))
    assert_connector_ok(result)