from .conftest import assert_connector_ok, assert_ingest_ok, dp


def test_ingest_csv(sample_csv, out_dir):
    result = dp("ingest", "--source", "local_file", "--path", str(sample_csv), "--batch-size", "30", out_dir=out_dir)
    assert_ingest_ok(result)


def test_ingest_parquet(sample_parquet, out_dir):
    result = dp("ingest", "--source", "local_file", "--path", str(sample_parquet), "--batch-size", "50", out_dir=out_dir)
    assert_ingest_ok(result)


def test_connector_test(sample_csv):
    result = dp("connector", "test", "--source", "local_file", "--path", str(sample_csv))
    assert_connector_ok(result)


def test_connector_schema(sample_csv):
    result = dp("connector", "schema", "--source", "local_file", "--path", str(sample_csv))
    assert_connector_ok(result)