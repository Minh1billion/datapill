import json
import socket
import subprocess
import tempfile
from pathlib import Path

import polars as pl
import pytest

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="session")
def work_dir():
    with tempfile.TemporaryDirectory(prefix="dp_test_") as tmp:
        yield Path(tmp)


@pytest.fixture(scope="session")
def data_csv():
    return _FIXTURES_DIR / "data.csv"


@pytest.fixture(scope="session")
def sample_csv(data_csv, work_dir):
    return data_csv


@pytest.fixture(scope="session")
def sample_parquet(data_csv, work_dir):
    path = work_dir / "sample.parquet"
    pl.read_csv(data_csv).write_parquet(path)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture(scope="session")
def out_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("artifacts")


@pytest.fixture(scope="session")
def pg_config(work_dir):
    return _write_json(work_dir / "pg.json", {
        "host": "localhost", "port": 5433,
        "database": "testdb", "user": "testuser", "password": "testpass",
    })


@pytest.fixture(scope="session")
def pg_query_config(work_dir):
    return _write_json(work_dir / "pg_query.json", {
        "host": "localhost", "port": 5433,
        "database": "testdb", "user": "testuser", "password": "testpass",
        "sql": "SELECT * FROM test_table LIMIT 20",
    })


@pytest.fixture(scope="session")
def my_config(work_dir):
    return _write_json(work_dir / "my.json", {
        "host": "localhost", "port": 3307,
        "database": "testdb", "user": "testuser", "password": "testpass",
    })


@pytest.fixture(scope="session")
def s3_config(work_dir):
    return _write_json(work_dir / "s3.json", {
        "aws_access_key_id": "minioadmin",
        "aws_secret_access_key": "minioadmin",
        "region": "us-east-1",
        "endpoint_url": "http://localhost:9000",
        "bucket": "testbucket",
    })


@pytest.fixture(scope="session")
def kafka_config(work_dir):
    return _write_json(work_dir / "kafka.json", {
        "bootstrap_servers": "localhost:9092",
        "group_id": "dp-test",
        "auto_offset_reset": "earliest",
        "default_topic": "dp-test-topic",
    })


@pytest.fixture(scope="session")
def rest_config(work_dir):
    return _write_json(work_dir / "rest.json", {
        "base_url": "https://jsonplaceholder.typicode.com",
        "response_path": "",
        "default_endpoint": "/todos",
    })


@pytest.fixture(scope="session")
def run_config(work_dir, sample_parquet):
    return _write_json(work_dir / "run.json", {
        "source": "local_file",
        "connector": {},
        "query": {"path": str(sample_parquet)},
        "ingest": {"batch_size": 50},
    })


@pytest.fixture(scope="session", autouse=True)
def pg_test_table():
    if not _port_open("localhost", 5433):
        return
    subprocess.run(
        ["docker", "exec", "-i",
         subprocess.check_output(["docker", "ps", "-qf", "name=postgres"]).decode().strip(),
         "psql", "-U", "testuser", "-d", "testdb"],
        input=(
            "CREATE TABLE IF NOT EXISTS test_table "
            "(id SERIAL PRIMARY KEY, name TEXT);\n"
            "INSERT INTO test_table (name) VALUES ('test1'),('test2'),('test3') "
            "ON CONFLICT DO NOTHING;\n"
        ),
        capture_output=True, text=True,
    )


def dp(*args, out_dir=None):
    import os
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    cmd = ["dp", *args]
    if out_dir:
        cmd += ["--out", str(out_dir)]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)


def assert_ingest_ok(result):
    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"ingest failed (exit {result.returncode})\n{combined[:800]}"


def assert_connector_ok(result):
    combined = result.stdout + result.stderr
    assert _matches(combined, r"Connection OK|Column|Schema:"), (
        f"connector check failed\n{combined[:500]}"
    )


def skip_if_down(host, port):
    if not _port_open(host, port):
        pytest.skip(f"{host}:{port} not reachable")


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


def _port_open(host, port, timeout=2):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _matches(text, pattern):
    import re
    return bool(re.search(pattern, text, re.IGNORECASE))