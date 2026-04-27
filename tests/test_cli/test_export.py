import json
import re
import socket

import polars as pl
import pytest

from .conftest import assert_ingest_ok, dp, skip_if_down


def _assert_export_ok(result):
    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"export failed (exit {result.returncode})\n{combined[:800]}"
    assert re.search(r"Destination|Rows|Dry run", combined, re.IGNORECASE), (
        f"export output missing expected fields\n{combined[:500]}"
    )


def _ingest_parquet(sample_parquet, out_dir) -> str:
    result = dp(
        "ingest", "--source", "local_file",
        "--path", str(sample_parquet),
        "--batch-size", "50",
        out_dir=out_dir,
    )
    assert_ingest_ok(result)
    combined = result.stdout + result.stderr
    m = re.search(r"Artifact:\s+(\S+)", combined)
    assert m, f"Could not find artifact ID in ingest output\n{combined[:500]}"
    return m.group(1)


@pytest.fixture(scope="module")
def parquet_from_csv(data_csv, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export_fmt")
    path = tmp / "input.parquet"
    pl.read_csv(data_csv).write_parquet(path)
    yield path
    if path.exists():
        path.unlink()


def test_export_parquet(parquet_from_csv, out_dir, tmp_path):
    artifact_id = _ingest_parquet(parquet_from_csv, out_dir)
    dest = tmp_path / "out.parquet"
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "parquet",
        "--out-path", str(dest),
        out_dir=out_dir,
    )
    _assert_export_ok(result)
    assert dest.exists()


def test_export_csv(parquet_from_csv, out_dir, tmp_path):
    artifact_id = _ingest_parquet(parquet_from_csv, out_dir)
    dest = tmp_path / "out.csv"
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "csv",
        "--out-path", str(dest),
        out_dir=out_dir,
    )
    _assert_export_ok(result)
    assert dest.exists()


def test_export_json(parquet_from_csv, out_dir, tmp_path):
    artifact_id = _ingest_parquet(parquet_from_csv, out_dir)
    dest = tmp_path / "out.json"
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "json",
        "--out-path", str(dest),
        out_dir=out_dir,
    )
    _assert_export_ok(result)
    assert dest.exists()


def test_export_jsonl(parquet_from_csv, out_dir, tmp_path):
    artifact_id = _ingest_parquet(parquet_from_csv, out_dir)
    dest = tmp_path / "out.jsonl"
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "jsonl",
        "--out-path", str(dest),
        out_dir=out_dir,
    )
    _assert_export_ok(result)
    assert dest.exists()


def test_export_excel(parquet_from_csv, out_dir, tmp_path):
    artifact_id = _ingest_parquet(parquet_from_csv, out_dir)
    dest = tmp_path / "out.xlsx"
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "excel",
        "--out-path", str(dest),
        out_dir=out_dir,
    )
    _assert_export_ok(result)
    assert dest.exists()


def test_export_parquet_with_compression(parquet_from_csv, out_dir, tmp_path):
    artifact_id = _ingest_parquet(parquet_from_csv, out_dir)
    dest = tmp_path / "out_snappy.parquet"
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "parquet",
        "--out-path", str(dest),
        "--compression", "snappy",
        out_dir=out_dir,
    )
    _assert_export_ok(result)
    assert dest.exists()


def test_export_from_parquet_file_directly(parquet_from_csv, out_dir, tmp_path):
    dest = tmp_path / "direct.parquet"
    result = dp(
        "export",
        "--input", str(parquet_from_csv),
        "--format", "parquet",
        "--out-path", str(dest),
        out_dir=out_dir,
    )
    _assert_export_ok(result)
    assert dest.exists()


def test_export_dry_run(parquet_from_csv, out_dir, tmp_path):
    artifact_id = _ingest_parquet(parquet_from_csv, out_dir)
    dest = tmp_path / "dry.parquet"
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "parquet",
        "--out-path", str(dest),
        "--dry-run",
        out_dir=out_dir,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined[:800]
    assert re.search(r"dry run", combined, re.IGNORECASE), (
        f"dry-run flag not reflected in output\n{combined[:500]}"
    )
    assert not dest.exists()


_PG_TYPE_MAP = {
    "Int64": "BIGINT",
    "Int32": "INTEGER",
    "Int16": "SMALLINT",
    "Int8": "SMALLINT",
    "Float64": "DOUBLE PRECISION",
    "Float32": "REAL",
    "String": "TEXT",
    "Boolean": "BOOLEAN",
    "Date": "DATE",
    "Datetime": "TIMESTAMP",
}


def _csv_to_pg_ddl(data_csv, table: str) -> str:
    df = pl.read_csv(data_csv, n_rows=0)
    cols = []
    for name, dtype in zip(df.columns, df.dtypes):
        pg_type = _PG_TYPE_MAP.get(str(dtype), "TEXT")
        pk = " PRIMARY KEY" if name == "id" else ""
        cols.append(f"  {name} {pg_type}{pk}")
    return f"DROP TABLE IF EXISTS {table};\nCREATE TABLE {table} (\n" + ",\n".join(cols) + "\n);\n"


@pytest.fixture(scope="session")
def pg_export_table(data_csv):
    try:
        socket.create_connection(("localhost", 5433), timeout=2).close()
    except OSError:
        return
    import subprocess
    container = subprocess.check_output(
        ["docker", "ps", "-qf", "name=postgres"], text=True
    ).strip()
    ddl = _csv_to_pg_ddl(data_csv, "export_test")
    subprocess.run(
        ["docker", "exec", "-i", container, "psql", "-U", "testuser", "-d", "testdb"],
        input=ddl,
        capture_output=True, text=True,
    )


@pytest.fixture
def pg_write_config(tmp_path, pg_export_table):
    cfg = {
        "source": "postgresql",
        "host": "localhost", "port": 5433,
        "database": "testdb", "user": "testuser", "password": "testpass",
        "table": "export_test",
    }
    p = tmp_path / "pg_write.json"
    p.write_text(json.dumps(cfg))
    return p


def test_export_writeback_postgresql(parquet_from_csv, out_dir, pg_write_config):
    skip_if_down("localhost", 5433)
    artifact_id = _ingest_parquet(parquet_from_csv, out_dir)
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "parquet",
        "--connector", str(pg_write_config),
        "--write-mode", "replace",
        out_dir=out_dir,
    )
    _assert_export_ok(result)


def test_export_upsert_postgresql(parquet_from_csv, out_dir, pg_write_config):
    skip_if_down("localhost", 5433)
    artifact_id = _ingest_parquet(parquet_from_csv, out_dir)
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "parquet",
        "--connector", str(pg_write_config),
        "--write-mode", "upsert",
        "--primary-keys", "id",
        out_dir=out_dir,
    )
    _assert_export_ok(result)


def test_export_missing_out_path_fails(parquet_from_csv, out_dir):
    artifact_id = _ingest_parquet(parquet_from_csv, out_dir)
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "parquet",
        out_dir=out_dir,
    )
    assert result.returncode != 0


def test_export_missing_input_flag(out_dir, tmp_path):
    dest = tmp_path / "out.parquet"
    result = dp(
        "export",
        "--format", "parquet",
        "--out-path", str(dest),
        out_dir=out_dir,
    )
    assert result.returncode != 0


def test_export_connector_missing_source_field(parquet_from_csv, out_dir, tmp_path):
    artifact_id = _ingest_parquet(parquet_from_csv, out_dir)
    bad_cfg = tmp_path / "bad_connector.json"
    bad_cfg.write_text(json.dumps({"host": "localhost"}))
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "parquet",
        "--connector", str(bad_cfg),
        out_dir=out_dir,
    )
    assert result.returncode != 0