import polars as pl
import pytest

from dataprep.features.preprocess.pipeline import PreprocessPipeline
from dataprep.features.preprocess.schema import StepConfig


def test_full_pipeline(csv_pipeline):
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
        StepConfig(step="impute_mode", columns=["category"]),
        StepConfig(step="clip_iqr", columns=["score"]),
        StepConfig(step="standard_scaler", columns=["score"]),
    ])
    report = pipeline.run(csv_pipeline)

    assert len(report.steps) == 4
    assert report.dry_run is False


def test_dry_run_returns_preview(csv_pipeline):
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
    ])
    report = pipeline.run(csv_pipeline, dry_run=True)

    assert report.dry_run is True
    assert report.preview_rows is not None
    assert len(report.preview_rows) <= 10
    assert report.output_schema is not None
    assert "score" in report.output_schema


def test_dry_run_no_checkpoint(csv_pipeline):
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
    ], checkpoint=True)
    report = pipeline.run(csv_pipeline, dry_run=True)

    assert report.checkpoints == []


def test_checkpoint_collects_dataframes(csv_pipeline):
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
        StepConfig(step="clip_iqr", columns=["score"]),
    ], checkpoint=True)
    report = pipeline.run(csv_pipeline)

    assert len(report.checkpoints) == 2
    for ckpt in report.checkpoints:
        assert isinstance(ckpt, pl.DataFrame)


def test_conflict_impute_then_drop_missing(csv_pipeline):
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
        StepConfig(step="drop_missing", columns=["score"]),
    ])
    report = pipeline.run(csv_pipeline)

    assert any("redundant" in w for w in report.warnings)


def test_conflict_clip_iqr_and_zscore_same_col(csv_pipeline):
    pipeline = PreprocessPipeline([
        StepConfig(step="clip_iqr", columns=["score"]),
        StepConfig(step="clip_zscore", columns=["score"]),
    ])
    report = pipeline.run(csv_pipeline)

    assert any("overlap" in w for w in report.warnings)


def test_numeric_step_on_string_raises(csv_pipeline):
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["category"]),
    ])
    with pytest.raises(TypeError, match="numeric"):
        pipeline.run(csv_pipeline)


def test_unknown_step_raises(csv_pipeline):
    pipeline = PreprocessPipeline([
        StepConfig(step="explode_everything"),
    ])
    with pytest.raises(ValueError, match="Unknown step"):
        pipeline.run(csv_pipeline)


def test_resume_from_step(csv_pipeline):
    pipeline = PreprocessPipeline([
        StepConfig(step="impute_mean", columns=["score"]),
        StepConfig(step="clip_iqr", columns=["score"]),
        StepConfig(step="standard_scaler", columns=["score"]),
    ], checkpoint=True)
    pipeline.run(csv_pipeline)
    report = pipeline.resume(csv_pipeline, from_step=2)

    assert len(report.steps) == 1