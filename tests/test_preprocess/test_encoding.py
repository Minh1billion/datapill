import polars as pl
import pytest

from dataprep.features.preprocess.schema import StepConfig
from dataprep.features.preprocess.steps.encoding import OneHotEncoder, OrdinalEncoder


@pytest.fixture
def df():
    return pl.DataFrame({"color": ["red", "blue", "red", "green"]})


def test_onehot(df):
    cfg = StepConfig(step="onehot", columns=["color"])
    result, stats = OneHotEncoder(cfg).apply(df)
    assert "color" not in result.columns
    assert "color__red" in result.columns
    assert "color__blue" in result.columns
    assert "color__green" in result.columns
    assert result["color__red"].to_list() == [1, 0, 1, 0]


def test_ordinal_auto(df):
    cfg = StepConfig(step="ordinal", columns=["color"])
    result, _ = OrdinalEncoder(cfg).apply(df)
    assert result["color"].dtype == pl.Int32
    vals = result["color"].to_list()
    assert len(set(vals)) == 3


def test_ordinal_with_order(df):
    cfg = StepConfig(step="ordinal", columns=["color"], params={"order": {"color": ["blue", "green", "red"]}})
    result, _ = OrdinalEncoder(cfg).apply(df)
    assert result["color"].to_list() == [2, 0, 2, 1]