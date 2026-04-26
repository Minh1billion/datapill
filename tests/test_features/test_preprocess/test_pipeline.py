import polars as pl
import pytest

from dataprep.features.preprocess.pipeline import PreprocessPipeline
from dataprep.features.preprocess.schema import StepConfig


@pytest.fixture
def df():
    return pl.DataFrame({
        "score": [1.0, None, 3.0, None, 5.0, 100.0],
        "category": ["a", "b", "a", "b", "a", None],
        "flag": [True, False, True, False, True, False],
    })


def test_full_pipeline(df):
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
        StepConfig(step="impute_mode", columns=["category"]),
        StepConfig(step="clip_iqr", columns=["score"]),
        StepConfig(step="standard_scaler", columns=["score"]),
    ])
    report = pipeline.run(df)
    assert len(report.steps) == 4
    assert report.dry_run is False


def test_dry_run_returns_preview(df):
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
    ])
    report = pipeline.run(df, dry_run=True)
    assert report.dry_run is True
    assert report.preview_rows is not None
    assert len(report.preview_rows) <= 10
    assert report.output_schema is not None
    assert "score" in report.output_schema


def test_dry_run_no_artifact(tmp_path, df, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
    ], checkpoint=True)
    pipeline.run(df, dry_run=True)
    assert not any(tmp_path.rglob("*.parquet"))


def test_checkpoint_saves_files(tmp_path, df, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
        StepConfig(step="clip_iqr", columns=["score"]),
    ], checkpoint=True)
    pipeline.run(df)
    checkpoints = list(tmp_path.rglob("*checkpoint*.parquet"))
    assert len(checkpoints) == 2


def test_conflict_impute_then_drop_missing(df):
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
        StepConfig(step="drop_missing", columns=["score"]),
    ])
    report = pipeline.run(df)
    assert any("redundant" in w for w in report.warnings)


def test_conflict_clip_iqr_and_zscore_same_col(df):
    pipeline = PreprocessPipeline([
        StepConfig(step="clip_iqr", columns=["score"]),
        StepConfig(step="clip_zscore", columns=["score"]),
    ])
    report = pipeline.run(df)
    assert any("overlap" in w for w in report.warnings)


def test_numeric_step_on_string_raises(df):
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["category"]),
    ])
    with pytest.raises(TypeError, match="numeric"):
        pipeline.run(df)


def test_unknown_step_raises(df):
    pipeline = PreprocessPipeline([
        StepConfig(step="explode_everything"),
    ])
    with pytest.raises(ValueError, match="Unknown step"):
        pipeline.run(df)


def test_resume_from_checkpoint(tmp_path, df, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
        StepConfig(step="clip_iqr", columns=["score"]),
        StepConfig(step="standard_scaler", columns=["score"]),
    ], checkpoint=True)
    pipeline.run(df)

    report = pipeline.resume(df, from_step=2)
    assert len(report.steps) == 1