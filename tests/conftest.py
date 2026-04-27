import json
import subprocess
import time
from pathlib import Path

import polars as pl
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATA_CSV = FIXTURES_DIR / "data.csv"

COMPOSE_FILE = "docker-compose.test.yml"

_HEALTHY_SERVICES = ["postgres", "mysql", "minio"]
_HEALTH_TIMEOUT = 120
_HEALTH_POLL = 3


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: mark test as requiring Docker services"
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    _docker_up()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "down", "-v"],
        capture_output=True,
    )


def _compose(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, *args],
        capture_output=True,
        text=True,
    )


def _container_id(service: str) -> str | None:
    result = _compose("ps", "-q", service)
    cid = result.stdout.strip()
    return cid if cid else None


def _health_status(container_id: str) -> str:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{json .State.Health}}", container_id],
        capture_output=True,
        text=True,
    )
    raw = result.stdout.strip()
    if not raw or raw == "null":
        return "none"
    try:
        return json.loads(raw).get("Status", "none")
    except json.JSONDecodeError:
        return "none"


def _wait_healthy(service: str) -> None:
    deadline = time.monotonic() + _HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        cid = _container_id(service)
        if cid and _health_status(cid) == "healthy":
            return
        time.sleep(_HEALTH_POLL)

    cid = _container_id(service)
    status = _health_status(cid) if cid else "container not found"
    logs = _compose("logs", "--tail", "30", service).stdout
    raise RuntimeError(
        f"Service '{service}' not healthy after {_HEALTH_TIMEOUT}s "
        f"(status={status!r}).\nLogs:\n{logs}"
    )


def _wait_minio_init() -> None:
    deadline = time.monotonic() + _HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        result = _compose("ps", "-a", "--format", "json", "minio-init")
        for line in result.stdout.strip().splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            exit_code = data.get("ExitCode", -1)
            state = data.get("State", "")
            if state == "exited" and exit_code == 0:
                return
            if state == "exited" and exit_code != 0:
                logs = _compose("logs", "minio-init").stdout
                raise RuntimeError(
                    f"minio-init exited with code {exit_code}.\nLogs:\n{logs}"
                )
        time.sleep(_HEALTH_POLL)

    raise RuntimeError(f"minio-init did not finish within {_HEALTH_TIMEOUT}s.")


def _docker_up() -> None:
    result = _compose("up", "-d")
    if result.returncode != 0:
        raise RuntimeError(
            f"docker compose up failed (rc={result.returncode}).\n"
            f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
    for svc in _HEALTHY_SERVICES:
        _wait_healthy(svc)
    _wait_minio_init()


@pytest.fixture(scope="session")
def sample_df() -> pl.DataFrame:
    return pl.read_csv(DATA_CSV, try_parse_dates=True)