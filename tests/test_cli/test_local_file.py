import polars as pl
import pytest

from .conftest import assert_connector_ok, assert_ingest_ok, dp


@pytest.fixture(scope="module")
def parquet_from_csv(data_csv, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("local_file_fmt")
    path = tmp / "input.parquet"
    pl.read_csv(data_csv).write_parquet(path)
    yield path
    if path.exists():
        path.unlink()


def test_ingest_csv(data_csv, out_dir):
    result = dp("ingest", "--source", "local_file", "--path", str(data_csv), "--batch-size", "30", out_dir=out_dir)
    assert_ingest_ok(result)


def test_ingest_parquet(parquet_from_csv, out_dir):
    result = dp("ingest", "--source", "local_file", "--path", str(parquet_from_csv), "--batch-size", "50", out_dir=out_dir)
    assert_ingest_ok(result)


def test_connector_test(data_csv):
    result = dp("connector", "test", "--source", "local_file", "--path", str(data_csv))
    assert_connector_ok(result)


def test_connector_schema(data_csv):
    result = dp("connector", "schema", "--source", "local_file", "--path", str(data_csv))
    assert_connector_ok(result)