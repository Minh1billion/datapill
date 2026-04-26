import os
import re
import textwrap
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import psutil
from RestrictedPython import (
    compile_restricted_exec,
    safe_builtins,
    safe_globals,
    utility_builtins,
)
from RestrictedPython.Guards import safer_getattr, full_write_guard

from .validator import analyze_ast, validate_schema, ASTAnalysisResult, SchemaValidationResult


SAMPLE_ROWS: int = 1_000
PREVIEW_ROWS: int = 100
DEFAULT_TIMEOUT_S: float = 30.0
DEFAULT_MEMORY_LIMIT_MB: float = 512.0

_PROCESS = psutil.Process(os.getpid())


@dataclass
class ResourceStats:
    execution_time_s: float
    memory_delta_mb: float

    @property
    def memory_peak_mb(self) -> float:
        return self.memory_delta_mb


@dataclass
class SandboxResult:
    ok: bool

    ast_result: ASTAnalysisResult | None = None
    schema_result: SchemaValidationResult | None = None
    resource_stats: ResourceStats | None = None

    preview: pl.DataFrame | None = None
    schema_diff: dict[str, Any] = field(default_factory=dict)

    failed_step: int | None = None
    error_message: str = ""
    traceback_str: str = ""


@dataclass
class SandboxConfig:
    timeout_s: float = DEFAULT_TIMEOUT_S
    memory_limit_mb: float = DEFAULT_MEMORY_LIMIT_MB
    allow_add_columns: bool = True
    allow_remove_columns: bool = False
    allow_dtype_change: bool = True


def _build_restricted_globals(df: pl.DataFrame) -> dict[str, Any]:
    glb: dict[str, Any] = safe_globals.copy()

    restricted_builtins = safe_builtins.copy()
    restricted_builtins.update(utility_builtins)
    glb["__builtins__"] = restricted_builtins

    glb["_getattr_"] = safer_getattr
    glb["_getitem_"] = lambda obj, key: obj[key]
    glb["_getiter_"] = iter
    glb["_write_"] = full_write_guard
    glb["_inplacevar_"] = _inplace_op

    glb["pl"] = pl
    glb["df"] = df

    return glb


def _inplace_op(op: str, x: Any, y: Any) -> Any:
    ops = {
        "+=": lambda a, b: a + b,
        "-=": lambda a, b: a - b,
        "*=": lambda a, b: a * b,
        "/=": lambda a, b: a / b,
        "//=": lambda a, b: a // b,
        "%=": lambda a, b: a % b,
        "**=": lambda a, b: a ** b,
    }
    fn = ops.get(op)
    if fn is None:
        raise NotImplementedError(f"In-place operator '{op}' not supported in sandbox")
    return fn(x, y)


class _TimeoutError(RuntimeError):
    pass


class _MemoryError(RuntimeError):
    pass


def _rss_mb() -> float:
    try:
        return _PROCESS.memory_info().rss / (1024 * 1024)
    except psutil.Error:
        return 0.0


def _run_restricted(
    code: str,
    df: pl.DataFrame,
    timeout_s: float,
    memory_limit_mb: float,
) -> tuple[pl.DataFrame, ResourceStats]:
    source = textwrap.dedent(code)

    compile_result = compile_restricted_exec(source, filename="<user_code>")
    if compile_result.errors:
        raise SyntaxError(
            "RestrictedPython compilation failed:\n"
            + "\n".join(compile_result.errors)
        )

    glb = _build_restricted_globals(df)

    result_holder: dict[str, Any] = {}
    exc_holder: dict[str, Any] = {}
    mem_holder: dict[str, float] = {}

    def _target() -> None:
        try:
            mem_holder["before"] = _rss_mb()
            exec(compile_result.code, glb)
            transform_fn = glb.get("transform")
            if transform_fn is None:
                raise ValueError(
                    "User code must define a function named 'transform(df)'."
                )
            output = transform_fn(df)
            if not isinstance(output, pl.DataFrame):
                raise TypeError(
                    f"'transform' must return a polars.DataFrame, "
                    f"got {type(output).__name__}"
                )
            mem_holder["after"] = _rss_mb()
            result_holder["output"] = output
        except Exception as exc:
            exc_holder["exc"] = exc
            exc_holder["tb"] = traceback.format_exc()

    t0 = time.perf_counter()
    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    elapsed = time.perf_counter() - t0

    if thread.is_alive():
        raise _TimeoutError(
            f"Execution exceeded timeout of {timeout_s:.1f}s"
        )

    if "exc" in exc_holder:
        raise exc_holder["exc"].with_traceback(None) from exc_holder["exc"]

    mem_before = mem_holder.get("before", 0.0)
    mem_after = mem_holder.get("after", 0.0)
    mem_delta = max(0.0, mem_after - mem_before)

    if mem_delta > memory_limit_mb:
        raise _MemoryError(
            f"Memory increase {mem_delta:.1f} MB exceeded limit {memory_limit_mb:.1f} MB"
        )

    stats = ResourceStats(execution_time_s=elapsed, memory_delta_mb=mem_delta)
    return result_holder["output"], stats


def run_sample(
    code: str,
    df: pl.DataFrame,
    cfg: SandboxConfig | None = None,
) -> SandboxResult:
    cfg = cfg or SandboxConfig()
    sample = df.head(SAMPLE_ROWS)

    try:
        ast_result = analyze_ast(code)
    except SyntaxError as exc:
        return SandboxResult(
            ok=False,
            failed_step=1,
            error_message=f"Syntax error: {exc}",
        )

    if not ast_result.ok:
        violations = "; ".join(str(v) for v in ast_result.violations)
        return SandboxResult(
            ok=False,
            ast_result=ast_result,
            failed_step=1,
            error_message=f"AST policy violations: {violations}",
        )

    try:
        output_sample, stats = _run_restricted(
            code, sample, cfg.timeout_s, cfg.memory_limit_mb
        )
    except _TimeoutError as exc:
        return SandboxResult(
            ok=False,
            ast_result=ast_result,
            failed_step=2,
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return SandboxResult(
            ok=False,
            ast_result=ast_result,
            failed_step=2,
            error_message=f"Execution error: {exc}",
            traceback_str=traceback.format_exc(),
        )

    schema_result = validate_schema(
        sample,
        output_sample,
        allow_add_columns=cfg.allow_add_columns,
        allow_remove_columns=cfg.allow_remove_columns,
        allow_dtype_change=cfg.allow_dtype_change,
    )

    if not schema_result.ok:
        violations = "; ".join(str(v) for v in schema_result.violations)
        return SandboxResult(
            ok=False,
            ast_result=ast_result,
            schema_result=schema_result,
            resource_stats=stats,
            failed_step=3,
            error_message=f"Schema validation failed: {violations}",
        )

    if stats.execution_time_s > cfg.timeout_s:
        return SandboxResult(
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

    if stats.memory_delta_mb > cfg.memory_limit_mb:
        return SandboxResult(
            ok=False,
            ast_result=ast_result,
            schema_result=schema_result,
            resource_stats=stats,
            failed_step=4,
            error_message=(
                f"Memory increase {stats.memory_delta_mb:.1f} MB "
                f"exceeded limit {cfg.memory_limit_mb:.1f} MB"
            ),
        )

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

    return SandboxResult(
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
    cfg: SandboxConfig | None = None,
) -> pl.DataFrame:
    cfg = cfg or SandboxConfig()

    try:
        output_df, _ = _run_restricted(
            code, df, cfg.timeout_s, cfg.memory_limit_mb
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        raise RuntimeError(
            f"Full execution failed — rolling back to original DataFrame.\n"
            f"Cause: {exc}\n\n{tb}"
        ) from exc

    return output_df


_SQL_BANNED_KEYWORDS: frozenset[str] = frozenset({
    "insert", "update", "delete", "drop", "create",
    "alter", "truncate", "replace", "merge", "call",
    "execute", "grant", "revoke", "attach", "detach",
    "pragma", "vacuum", "analyze",
})


def run_sql(query: str, df: pl.DataFrame) -> pl.DataFrame:
    import duckdb

    normalized = query.strip().lower()

    if not (normalized.startswith("select") or normalized.startswith("with")):
        raise ValueError(
            "Only SELECT statements are allowed in SQL custom steps. "
            f"Query starts with: '{query.strip()[:40]}'"
        )

    for kw in _SQL_BANNED_KEYWORDS:
        if re.search(rf"\b{kw}\b", normalized):
            raise ValueError(
                f"SQL keyword '{kw.upper()}' is not allowed in custom SQL steps."
            )

    try:
        con = duckdb.connect()
        con.register("df", df)
        result: pl.DataFrame = con.execute(query).pl()
        con.close()
        return result
    except duckdb.Error as exc:
        raise RuntimeError(f"DuckDB execution error: {exc}") from exc