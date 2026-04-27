import polars as pl
import pytest

from dataprep.features.preprocess.schema import StepConfig
from dataprep.features.preprocess.steps.scaling import MinMaxScaler, RobustScaler, StandardScaler


def test_standard_scaler(csv_scaling):
    col = "score"
    cfg = StepConfig(step="standard_scaler", columns=[col])
    result, _ = StandardScaler(cfg).apply(csv_scaling)

    assert result[col].mean() == pytest.approx(0.0, abs=1e-9)
    assert result[col].std() == pytest.approx(1.0, abs=1e-6)


def test_minmax_scaler(csv_scaling):
    col = "score"
    cfg = StepConfig(step="minmax_scaler", columns=[col])
    result, _ = MinMaxScaler(cfg).apply(csv_scaling)

    assert result[col].min() == pytest.approx(0.0)
    assert result[col].max() == pytest.approx(1.0)


def test_robust_scaler(csv_scaling):
    col = "score"
    cfg = StepConfig(step="robust_scaler", columns=[col])
    result, _ = RobustScaler(cfg).apply(csv_scaling)

    median = csv_scaling[col].median()
    q1 = csv_scaling[col].quantile(0.25)
    q3 = csv_scaling[col].quantile(0.75)
    iqr = q3 - q1

    assert result[col].median() == pytest.approx((median - median) / iqr, abs=1e-9)


def test_standard_scaler_zero_std(tmp_path_factory):
    p = tmp_path_factory.getbasetemp() / "data_zero_std.csv"
    pl.DataFrame({"x": [5.0, 5.0, 5.0]}).write_csv(p)
    df = pl.read_csv(p)

    cfg = StepConfig(step="standard_scaler", columns=["x"])
    result, _ = StandardScaler(cfg).apply(df)

    assert result["x"].to_list() == [5.0, 5.0, 5.0]
    p.unlink(missing_ok=True)