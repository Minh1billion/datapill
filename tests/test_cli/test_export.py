import json
import re
import socket

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


def test_export_parquet(sample_parquet, out_dir, tmp_path):
    artifact_id = _ingest_parquet(sample_parquet, out_dir)
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


def test_export_csv(sample_parquet, out_dir, tmp_path):
    artifact_id = _ingest_parquet(sample_parquet, out_dir)
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


def test_export_json(sample_parquet, out_dir, tmp_path):
    artifact_id = _ingest_parquet(sample_parquet, out_dir)
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


def test_export_jsonl(sample_parquet, out_dir, tmp_path):
    artifact_id = _ingest_parquet(sample_parquet, out_dir)
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


def test_export_excel(sample_parquet, out_dir, tmp_path):
    artifact_id = _ingest_parquet(sample_parquet, out_dir)
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


def test_export_parquet_with_compression(sample_parquet, out_dir, tmp_path):
    artifact_id = _ingest_parquet(sample_parquet, out_dir)
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


def test_export_from_parquet_file_directly(sample_parquet, out_dir, tmp_path):
    dest = tmp_path / "direct.parquet"
    result = dp(
        "export",
        "--input", str(sample_parquet),
        "--format", "parquet",
        "--out-path", str(dest),
        out_dir=out_dir,
    )
    _assert_export_ok(result)
    assert dest.exists()


def test_export_dry_run(sample_parquet, out_dir, tmp_path):
    artifact_id = _ingest_parquet(sample_parquet, out_dir)
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


@pytest.fixture(scope="session")
def pg_export_table():
    try:
        socket.create_connection(("localhost", 5433), timeout=2).close()
    except OSError:
        return
    import subprocess
    container = subprocess.check_output(
        ["docker", "ps", "-qf", "name=postgres"], text=True
    ).strip()
    subprocess.run(
        ["docker", "exec", "-i", container, "psql", "-U", "testuser", "-d", "testdb"],
        input=(
            "CREATE TABLE IF NOT EXISTS export_test ("
            "  id      INTEGER PRIMARY KEY,"
            "  name    TEXT,"
            "  score   DOUBLE PRECISION,"
            "  active  BOOLEAN"
            ");\n"
        ),
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


def test_export_writeback_postgresql(sample_parquet, out_dir, pg_write_config):
    skip_if_down("localhost", 5433)
    artifact_id = _ingest_parquet(sample_parquet, out_dir)
    result = dp(
        "export",
        "--input", artifact_id,
        "--format", "parquet",
        "--connector", str(pg_write_config),
        "--write-mode", "replace",
        out_dir=out_dir,
    )
    _assert_export_ok(result)


def test_export_upsert_postgresql(sample_parquet, out_dir, pg_write_config):
    skip_if_down("localhost", 5433)
    artifact_id = _ingest_parquet(sample_parquet, out_dir)
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


def test_export_missing_out_path_fails(sample_parquet, out_dir):
    artifact_id = _ingest_parquet(sample_parquet, out_dir)
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


def test_export_connector_missing_source_field(sample_parquet, out_dir, tmp_path):
    artifact_id = _ingest_parquet(sample_parquet, out_dir)
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