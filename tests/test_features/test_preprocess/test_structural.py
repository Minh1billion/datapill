import polars as pl
import pytest

from dataprep.features.preprocess.schema import StepConfig
from dataprep.features.preprocess.steps.structural import CastDtype, Deduplicate, DropColumns, RenameColumns, SelectColumns


@pytest.fixture
def df():
    return pl.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "z"], "c": [1.0, 2.0, 2.0, 3.0]})


def test_select_columns(df):
    cfg = StepConfig(step="select_columns", columns=["a", "b"])
    result, _ = SelectColumns(cfg).apply(df)
    assert result.columns == ["a", "b"]


def test_drop_columns(df):
    cfg = StepConfig(step="drop_columns", columns=["c"])
    result, _ = DropColumns(cfg).apply(df)
    assert "c" not in result.columns
    assert "a" in result.columns


def test_rename_columns(df):
    cfg = StepConfig(step="rename_columns", params={"mapping": {"a": "id", "b": "label"}})
    result, _ = RenameColumns(cfg).apply(df)
    assert "id" in result.columns
    assert "label" in result.columns
    assert "a" not in result.columns


def test_cast_dtype(df):
    cfg = StepConfig(step="cast_dtype", params={"casts": {"a": "float32"}})
    result, stats = CastDtype(cfg).apply(df)
    assert result["a"].dtype == pl.Float32
    assert "a" in stats.dtype_changes


def test_cast_dtype_unknown_raises(df):
    cfg = StepConfig(step="cast_dtype", params={"casts": {"a": "matrix"}})
    with pytest.raises(ValueError, match="Unknown dtype"):
        CastDtype(cfg).apply(df)


def test_deduplicate_all(df):
    cfg = StepConfig(step="deduplicate")
    result, stats = Deduplicate(cfg).apply(df)
    assert len(result) == 3
    assert stats.row_delta == -1


def test_deduplicate_subset(df):
    cfg = StepConfig(step="deduplicate", columns=["b"])
    result, stats = Deduplicate(cfg).apply(df)
    assert result["b"].n_unique() == len(result)