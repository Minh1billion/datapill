import json
import os
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .validator import analyze_ast, validate_schema, ASTAnalysisResult, SchemaValidationResult


DEFAULT_IMAGE: str = os.getenv(
    "DATAPREP_SANDBOX_IMAGE", "python:3.12-slim"
)
DEFAULT_TIMEOUT_S: float = 60.0
DEFAULT_MEMORY_LIMIT: str = "512m"
DEFAULT_CPU_LIMIT: str = "1.0"
SAMPLE_ROWS: int = 1_000
PREVIEW_ROWS: int = 100


_RUNNER_SCRIPT: str = textwrap.dedent("""\
    import sys, json
    import polars as pl

    df = pl.read_parquet("/sandbox/input.parquet")

    # Load and exec user code
    with open("/sandbox/code.py") as f:
        source = f.read()

    glb: dict = {"pl": pl, "df": df}
    exec(compile(source, "<user_code>", "exec"), glb)

    transform = glb.get("transform")
    if transform is None:
        print(json.dumps({"error": "No 'transform' function defined"}))
        sys.exit(1)

    try:
        output = transform(df)
        if not isinstance(output, pl.DataFrame):
            raise TypeError(f"transform must return pl.DataFrame, got {type(output).__name__}")
        output.write_parquet("/sandbox/output.parquet")
        print(json.dumps({"ok": True, "rows": len(output)}))
    except Exception as exc:
        import traceback
        print(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}))
        sys.exit(1)
""")


@dataclass
class DockerResourceStats:
    execution_time_s: float


@dataclass
class DockerSandboxResult:
    ok: bool

    ast_result: ASTAnalysisResult | None = None
    schema_result: SchemaValidationResult | None = None
    resource_stats: DockerResourceStats | None = None

    preview: pl.DataFrame | None = None
    schema_diff: dict[str, Any] = field(default_factory=dict)

    failed_step: int | None = None
    error_message: str = ""
    traceback_str: str = ""


@dataclass
class DockerRunnerConfig:
    image: str = DEFAULT_IMAGE
    timeout_s: float = DEFAULT_TIMEOUT_S
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    cpu_limit: str = DEFAULT_CPU_LIMIT
    allow_add_columns: bool = True
    allow_remove_columns: bool = False
    allow_dtype_change: bool = True


def _check_docker_available() -> None:
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Docker is not available. "
            "Install Docker or use the RestrictedPython sandbox instead."
        )


def _run_in_docker(
    code: str,
    df: pl.DataFrame,
    cfg: DockerRunnerConfig,
) -> tuple[pl.DataFrame, DockerResourceStats]:
    with tempfile.TemporaryDirectory(prefix="dataprep_sandbox_") as tmpdir:
        tmp = Path(tmpdir)

        df.write_parquet(tmp / "input.parquet")
        (tmp / "code.py").write_text(code, encoding="utf-8")
        (tmp / "runner.py").write_text(_RUNNER_SCRIPT, encoding="utf-8")

        cmd = [
            "docker", "run",
            "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:size=64m",
            "--memory", cfg.memory_limit,
            "--cpus", cfg.cpu_limit,
            "--user", "65534:65534",
            "--volume", f"{tmpdir}:/sandbox:rw",
            cfg.image,
            "python", "/sandbox/runner.py",
        ]

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=cfg.timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Docker execution exceeded timeout of {cfg.timeout_s:.1f}s"
            )
        elapsed = time.perf_counter() - t0

        stats = DockerResourceStats(execution_time_s=elapsed)

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            try:
                payload = json.loads(stdout)
                msg = payload.get("error", stderr)
                tb = payload.get("traceback", "")
            except json.JSONDecodeError:
                msg = stderr or stdout or "Unknown error"
                tb = ""
            raise RuntimeError(msg)

        try:
            payload = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            payload = {}

        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "Unknown runner error"))

        output_path = tmp / "output.parquet"
        if not output_path.exists():
            raise RuntimeError("Runner did not produce output.parquet")

        output_df = pl.read_parquet(output_path)
        return output_df, stats


def run_sample(
    code: str,
    df: pl.DataFrame,
    cfg: DockerRunnerConfig | None = None,
) -> DockerSandboxResult:
    cfg = cfg or DockerRunnerConfig()

    # Step 1: AST (still run locally - fast pre-filter before spinning container)
    try:
        ast_result = analyze_ast(code)
    except SyntaxError as exc:
        return DockerSandboxResult(
            ok=False,
            failed_step=1,
            error_message=f"Syntax error: {exc}",
        )

    if not ast_result.ok:
        violations = "; ".join(str(v) for v in ast_result.violations)
        return DockerSandboxResult(
            ok=False,
            ast_result=ast_result,
            failed_step=1,
            error_message=f"AST policy violations: {violations}",
        )

    sample = df.head(SAMPLE_ROWS)

    # Step 2: Docker execution
    try:
        output_sample, stats = _run_in_docker(code, sample, cfg)
    except TimeoutError as exc:
        return DockerSandboxResult(
            ok=False,
            ast_result=ast_result,
            failed_step=2,
            error_message=str(exc),
        )
    except Exception as exc:
        return DockerSandboxResult(
            ok=False,
            ast_result=ast_result,
            failed_step=2,
            error_message=f"Docker execution error: {exc}",
        )

    # Step 3: Schema validation
    schema_result = validate_schema(
        sample,
        output_sample,
        allow_add_columns=cfg.allow_add_columns,
        allow_remove_columns=cfg.allow_remove_columns,
        allow_dtype_change=cfg.allow_dtype_change,
    )

    if not schema_result.ok:
        violations = "; ".join(str(v) for v in schema_result.violations)
        return DockerSandboxResult(
            ok=False,
            ast_result=ast_result,
            schema_result=schema_result,
            resource_stats=stats,
            failed_step=3,
            error_message=f"Schema validation failed: {violations}",
        )

    # Step 4: Resource check
    if stats.execution_time_s > cfg.timeout_s:
        return DockerSandboxResult(
            ok=False,
            ast_result=ast_result,
            schema_result=schema_result,
            resource_stats=stats,
            failed_step=4,
            error_message=(
                f"Execution time {stats.execution_time_s:.2f}s "
                f"exceeded limit {cfg.timeout_s:.1f}s"
            ),
        )

    # Step 5: Preview + schema diff
    preview = output_sample.head(PREVIEW_ROWS)
    schema_diff: dict[str, Any] = {
        "added_columns": schema_result.added_columns,
        "removed_columns": schema_result.removed_columns,
        "changed_dtypes": [
            {"column": col, "before": before, "after": after}
            for col, before, after in schema_result.changed_dtypes
        ],
        "row_count_in": len(sample),
        "row_count_out": len(output_sample),
    }

    return DockerSandboxResult(
        ok=True,
        ast_result=ast_result,
        schema_result=schema_result,
        resource_stats=stats,
        preview=preview,
        schema_diff=schema_diff,
    )


def apply(
    code: str,
    df: pl.DataFrame,
    cfg: DockerRunnerConfig | None = None,
) -> pl.DataFrame:
    cfg = cfg or DockerRunnerConfig()

    try:
        output_df, _ = _run_in_docker(code, df, cfg)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Docker full execution failed - rolling back to original DataFrame.\n"
            f"Cause: {exc}"
        ) from exc

    return output_df