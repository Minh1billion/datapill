import polars as pl
import pytest

from dataprep.features.preprocess.schema import StepConfig
from dataprep.features.preprocess.steps.missing import DropMissing, ImputeMean, ImputeMedian, ImputeMode


@pytest.fixture
def df():
    return pl.DataFrame({
        "a": [1.0, None, 3.0, None, 5.0],
        "b": ["x", None, "x", "y", None],
    })


def test_impute_mean(df):
    cfg = StepConfig(step="impute_mean", columns=["a"])
    result, stats = ImputeMean(cfg).apply(df)
    assert result["a"].null_count() == 0
    assert result["a"][1] == pytest.approx((1.0 + 3.0 + 5.0) / 3)
    assert stats.null_delta["a"] == -2


def test_impute_median(df):
    cfg = StepConfig(step="impute_median", columns=["a"])
    result, stats = ImputeMedian(cfg).apply(df)
    assert result["a"].null_count() == 0
    assert result["a"][1] == pytest.approx(3.0)


def test_impute_mode(df):
    cfg = StepConfig(step="impute_mode", columns=["b"])
    result, stats = ImputeMode(cfg).apply(df)
    assert result["b"].null_count() == 0
    assert result["b"][1] == "x"


def test_drop_missing(df):
    cfg = StepConfig(step="drop_missing", columns=["a"])
    result, stats = DropMissing(cfg).apply(df)
    assert result["a"].null_count() == 0
    assert len(result) == 3
    assert stats.row_delta == -2