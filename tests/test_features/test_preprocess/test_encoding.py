import polars as pl
import pytest

from dataprep.features.preprocess.schema import StepConfig
from dataprep.features.preprocess.steps.encoding import OneHotEncoder, OrdinalEncoder


def test_onehot(csv_encoding):
    col = "category"
    cfg = StepConfig(step="onehot", columns=[col])
    result, stats = OneHotEncoder(cfg).apply(csv_encoding)

    unique_vals = csv_encoding[col].unique().to_list()

    assert col not in result.columns
    for v in unique_vals:
        assert f"{col}__{v}" in result.columns

    for v in unique_vals:
        expected = (csv_encoding[col] == v).cast(pl.Int8).to_list()
        assert result[f"{col}__{v}"].to_list() == expected


def test_ordinal_auto(csv_encoding):
    col = "category"
    cfg = StepConfig(step="ordinal", columns=[col])
    result, _ = OrdinalEncoder(cfg).apply(csv_encoding)

    assert result[col].dtype == pl.Int32
    n_unique = csv_encoding[col].n_unique()
    assert len(result[col].unique()) == n_unique


def test_ordinal_with_order(csv_encoding):
    col = "region"
    unique_vals = csv_encoding[col].unique().sort().to_list()
    cfg = StepConfig(step="ordinal", columns=[col], params={"order": {col: unique_vals}})
    result, _ = OrdinalEncoder(cfg).apply(csv_encoding)

    order_map = {v: i for i, v in enumerate(unique_vals)}
    expected = [order_map[v] for v in csv_encoding[col].to_list()]
    assert result[col].to_list() == expected