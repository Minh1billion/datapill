import json
import re
from pathlib import Path

import polars as pl
import pytest

from .conftest import dp


def _write_pipeline(tmp_path: Path, steps: list) -> Path:
    p = tmp_path / "pipeline.json"
    p.write_text(json.dumps({"steps": steps}))
    return p


def _assert_preprocess_ok(result):
    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"preprocess failed (exit {result.returncode})\n{combined[:800]}"
    assert re.search(r"Output|Run ID|Dry run", combined, re.IGNORECASE), (
        f"preprocess output missing expected fields\n{combined[:500]}"
    )


@pytest.fixture(scope="module")
def parquet_from_csv(data_csv, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("preprocess_fmt")
    path = tmp / "input.parquet"
    pl.read_csv(data_csv).write_parquet(path)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


@pytest.fixture
def impute_pipeline(tmp):
    df = pl.read_csv(pytest.importorskip("pathlib").Path(
        __import__("conftest", fromlist=["_FIXTURES_DIR"])._FIXTURES_DIR / "data.csv"
    )) if False else None
    return _write_pipeline(tmp, [
        {"type": "impute_mean", "scope": {"columns": ["score"]}},
    ])


@pytest.fixture
def multi_step_pipeline(tmp):
    return _write_pipeline(tmp, [
        {"type": "impute_mean",      "scope": {"columns": ["score"]}},
        {"type": "clip_iqr",         "scope": {"columns": ["score"]}},
        {"type": "standard_scaler",  "scope": {"columns": ["score"]}},
    ])


@pytest.fixture
def all_columns_pipeline(tmp, data_csv):
    df = pl.read_csv(data_csv)
    numeric_cols = [c for c, t in zip(df.columns, df.dtypes) if t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)]
    return _write_pipeline(tmp, [
        {"type": "impute_mean", "scope": {"columns": numeric_cols}},
    ])


def test_preprocess_single_step(parquet_from_csv, out_dir, impute_pipeline):
    result = dp(
        "preprocess",
        "--input", str(parquet_from_csv),
        "--pipeline", str(impute_pipeline),
        out_dir=out_dir,
    )
    _assert_preprocess_ok(result)


def test_preprocess_multi_step(parquet_from_csv, out_dir, multi_step_pipeline):
    result = dp(
        "preprocess",
        "--input", str(parquet_from_csv),
        "--pipeline", str(multi_step_pipeline),
        out_dir=out_dir,
    )
    _assert_preprocess_ok(result)


def test_preprocess_dry_run(parquet_from_csv, out_dir, multi_step_pipeline):
    result = dp(
        "preprocess",
        "--input", str(parquet_from_csv),
        "--pipeline", str(multi_step_pipeline),
        "--dry-run",
        out_dir=out_dir,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined[:800]
    assert re.search(r"dry run", combined, re.IGNORECASE), (
        f"dry-run flag not reflected in output\n{combined[:500]}"
    )
    assert not re.search(r"Output:\s+\S", combined), (
        f"dry run should not write an artifact\n{combined[:500]}"
    )


def test_preprocess_checkpoint(parquet_from_csv, out_dir, multi_step_pipeline):
    result = dp(
        "preprocess",
        "--input", str(parquet_from_csv),
        "--pipeline", str(multi_step_pipeline),
        "--checkpoint",
        out_dir=out_dir,
    )
    _assert_preprocess_ok(result)


def test_preprocess_all_columns(parquet_from_csv, out_dir, all_columns_pipeline):
    result = dp(
        "preprocess",
        "--input", str(parquet_from_csv),
        "--pipeline", str(all_columns_pipeline),
        out_dir=out_dir,
    )
    _assert_preprocess_ok(result)


def test_preprocess_empty_steps_fails(parquet_from_csv, out_dir, tmp):
    p = tmp / "empty_pipeline.json"
    p.write_text(json.dumps({"steps": []}))
    result = dp(
        "preprocess",
        "--input", str(parquet_from_csv),
        "--pipeline", str(p),
        out_dir=out_dir,
    )
    assert result.returncode != 0, "expected failure for empty steps"


def test_preprocess_missing_pipeline_file(parquet_from_csv, out_dir):
    result = dp(
        "preprocess",
        "--input", str(parquet_from_csv),
        "--pipeline", "/nonexistent/pipeline.json",
        out_dir=out_dir,
    )
    assert result.returncode != 0, "expected failure for missing pipeline file"


def test_preprocess_missing_input_flag(out_dir, impute_pipeline):
    result = dp(
        "preprocess",
        "--pipeline", str(impute_pipeline),
        out_dir=out_dir,
    )
    assert result.returncode != 0