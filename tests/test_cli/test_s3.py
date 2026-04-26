import pytest

from .conftest import assert_connector_ok, dp, skip_if_down


@pytest.fixture(autouse=True)
def require_minio():
    skip_if_down("localhost", 9000)


def test_connector_test(s3_config):
    result = dp("connector", "test", "--source", "s3", "--config", str(s3_config))
    assert_connector_ok(result)