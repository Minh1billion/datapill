import polars as pl
import pytest

from dataprep.features.preprocess.schema import StepConfig
from dataprep.features.preprocess.steps.outlier import ClipIQR, ClipZScore


@pytest.fixture
def df():
    return pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})


def test_clip_iqr(df):
    cfg = StepConfig(step="clip_iqr", columns=["val"])
    result, stats = ClipIQR(cfg).apply(df)
    q1 = df["val"].quantile(0.25)
    q3 = df["val"].quantile(0.75)
    iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    assert result["val"].max() <= hi
    assert result["val"][-1] < 100.0


def test_clip_zscore_default_threshold(df):
    cfg = StepConfig(step="clip_zscore", columns=["val"])
    result, stats = ClipZScore(cfg).apply(df)
    mean = df["val"].mean()
    std = df["val"].std()
    assert result["val"].max() <= mean + 3 * std


def test_clip_zscore_custom_threshold(df):
    cfg = StepConfig(step="clip_zscore", columns=["val"], params={"threshold": 1.0})
    result, stats = ClipZScore(cfg).apply(df)
    mean = df["val"].mean()
    std = df["val"].std()
    assert result["val"].max() <= mean + 1.0 * std