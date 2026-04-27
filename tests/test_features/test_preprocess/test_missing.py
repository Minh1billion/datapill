import polars as pl
import pytest

from dataprep.features.preprocess.schema import StepConfig
from dataprep.features.preprocess.steps.missing import DropMissing, ImputeMean, ImputeMedian, ImputeMode

_NULL_INDICES = [1, 3]


def test_impute_mean(csv_with_nulls):
    col = "score"
    cfg = StepConfig(step="impute_mean", columns=[col])
    result, stats = ImputeMean(cfg).apply(csv_with_nulls)

    non_null_mean = csv_with_nulls[col].drop_nulls().mean()
    expected_null_count = csv_with_nulls[col].null_count()

    assert result[col].null_count() == 0
    for i in _NULL_INDICES:
        assert result[col][i] == pytest.approx(non_null_mean)
    assert stats.null_delta[col] == -expected_null_count


def test_impute_median(csv_with_nulls):
    col = "score"
    cfg = StepConfig(step="impute_median", columns=[col])
    result, stats = ImputeMedian(cfg).apply(csv_with_nulls)

    non_null_median = csv_with_nulls[col].drop_nulls().median()

    assert result[col].null_count() == 0
    for i in _NULL_INDICES:
        assert result[col][i] == pytest.approx(non_null_median)


def test_impute_mode(csv_with_nulls):
    col = "category"
    cfg = StepConfig(step="impute_mode", columns=[col])
    result, stats = ImputeMode(cfg).apply(csv_with_nulls)

    mode_val = csv_with_nulls[col].drop_nulls().mode().sort()[0]

    assert result[col].null_count() == 0
    for i in _NULL_INDICES:
        assert result[col][i] == mode_val


def test_drop_missing(csv_with_nulls):
    col = "score"
    cfg = StepConfig(step="drop_missing", columns=[col])
    result, stats = DropMissing(cfg).apply(csv_with_nulls)

    expected_null_count = csv_with_nulls[col].null_count()
    expected_len = len(csv_with_nulls) - expected_null_count

    assert result[col].null_count() == 0
    assert len(result) == expected_len
    assert stats.row_delta == -expected_null_count