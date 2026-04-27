import polars as pl
import pytest

from dataprep.features.preprocess.schema import StepConfig
from dataprep.features.preprocess.steps.outlier import ClipIQR, ClipZScore


def test_clip_iqr(csv_with_outliers):
    col = "value"
    cfg = StepConfig(step="clip_iqr", columns=[col])
    result, stats = ClipIQR(cfg).apply(csv_with_outliers)

    q1 = csv_with_outliers[col].quantile(0.25)
    q3 = csv_with_outliers[col].quantile(0.75)
    iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    lo = q1 - 1.5 * iqr

    assert result[col].max() <= hi
    assert result[col].min() >= lo
    assert result[col].max() < csv_with_outliers[col].max()


def test_clip_zscore_default_threshold(csv_with_outliers):
    col = "value"
    cfg = StepConfig(step="clip_zscore", columns=[col])
    result, stats = ClipZScore(cfg).apply(csv_with_outliers)

    mean = csv_with_outliers[col].mean()
    std = csv_with_outliers[col].std()

    assert result[col].max() <= mean + 3 * std
    assert result[col].min() >= mean - 3 * std


def test_clip_zscore_custom_threshold(csv_with_outliers):
    col = "value"
    threshold = 1.0
    cfg = StepConfig(step="clip_zscore", columns=[col], params={"threshold": threshold})
    result, stats = ClipZScore(cfg).apply(csv_with_outliers)

    mean = csv_with_outliers[col].mean()
    std = csv_with_outliers[col].std()

    assert result[col].max() <= mean + threshold * std
    assert result[col].min() >= mean - threshold * std