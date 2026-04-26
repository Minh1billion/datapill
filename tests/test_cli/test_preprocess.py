import json
import re
import tempfile
from pathlib import Path

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


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


@pytest.fixture
def impute_pipeline(tmp):
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
def all_columns_pipeline(tmp):
    return _write_pipeline(tmp, [
        {"type": "impute_mean", "scope": {"columns": ["id", "score"]}},
    ])


def test_preprocess_single_step(sample_parquet, out_dir, impute_pipeline):
    result = dp(
        "preprocess",
        "--input", str(sample_parquet),
        "--pipeline", str(impute_pipeline),
        out_dir=out_dir,
    )
    _assert_preprocess_ok(result)


def test_preprocess_multi_step(sample_parquet, out_dir, multi_step_pipeline):
    result = dp(
        "preprocess",
        "--input", str(sample_parquet),
        "--pipeline", str(multi_step_pipeline),
        out_dir=out_dir,
    )
    _assert_preprocess_ok(result)


def test_preprocess_dry_run(sample_parquet, out_dir, multi_step_pipeline):
    result = dp(
        "preprocess",
        "--input", str(sample_parquet),
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


def test_preprocess_checkpoint(sample_parquet, out_dir, multi_step_pipeline):
    result = dp(
        "preprocess",
        "--input", str(sample_parquet),
        "--pipeline", str(multi_step_pipeline),
        "--checkpoint",
        out_dir=out_dir,
    )
    _assert_preprocess_ok(result)


def test_preprocess_all_columns(sample_parquet, out_dir, all_columns_pipeline):
    result = dp(
        "preprocess",
        "--input", str(sample_parquet),
        "--pipeline", str(all_columns_pipeline),
        out_dir=out_dir,
    )
    _assert_preprocess_ok(result)


def test_preprocess_empty_steps_fails(sample_parquet, out_dir, tmp):
    empty_pipeline = _write_pipeline(tmp / "empty.json" if False else tmp, [])
    p = tmp / "empty_pipeline.json"
    p.write_text(json.dumps({"steps": []}))
    result = dp(
        "preprocess",
        "--input", str(sample_parquet),
        "--pipeline", str(p),
        out_dir=out_dir,
    )
    assert result.returncode != 0, "expected failure for empty steps"


def test_preprocess_missing_pipeline_file(sample_parquet, out_dir):
    result = dp(
        "preprocess",
        "--input", str(sample_parquet),
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