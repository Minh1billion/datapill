import polars as pl
import pytest

from dataprep.features.preprocess.schema import StepConfig
from dataprep.features.preprocess.steps.scaling import MinMaxScaler, RobustScaler, StandardScaler


@pytest.fixture
def df():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})


def test_standard_scaler(df):
    cfg = StepConfig(step="standard_scaler", columns=["x"])
    result, _ = StandardScaler(cfg).apply(df)
    assert result["x"].mean() == pytest.approx(0.0, abs=1e-9)
    assert result["x"].std() == pytest.approx(1.0, abs=1e-6)


def test_minmax_scaler(df):
    cfg = StepConfig(step="minmax_scaler", columns=["x"])
    result, _ = MinMaxScaler(cfg).apply(df)
    assert result["x"].min() == pytest.approx(0.0)
    assert result["x"].max() == pytest.approx(1.0)


def test_robust_scaler(df):
    cfg = StepConfig(step="robust_scaler", columns=["x"])
    result, _ = RobustScaler(cfg).apply(df)
    median = df["x"].median()
    q1 = df["x"].quantile(0.25)
    q3 = df["x"].quantile(0.75)
    iqr = q3 - q1
    expected_median_scaled = (median - median) / iqr
    assert result["x"].median() == pytest.approx(expected_median_scaled, abs=1e-9)


def test_standard_scaler_zero_std():
    df = pl.DataFrame({"x": [5.0, 5.0, 5.0]})
    cfg = StepConfig(step="standard_scaler", columns=["x"])
    result, _ = StandardScaler(cfg).apply(df)
    assert result["x"].to_list() == [5.0, 5.0, 5.0]