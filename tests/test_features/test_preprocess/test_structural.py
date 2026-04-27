import polars as pl
import pytest

from dataprep.features.preprocess.schema import StepConfig
from dataprep.features.preprocess.steps.structural import CastDtype, Deduplicate, DropColumns, RenameColumns, SelectColumns


def test_select_columns(csv_structural):
    cols = ["id", "score"]
    cfg = StepConfig(step="select_columns", columns=cols)
    result, _ = SelectColumns(cfg).apply(csv_structural)

    assert result.columns == cols


def test_drop_columns(csv_structural):
    col = "tag"
    cfg = StepConfig(step="drop_columns", columns=[col])
    result, _ = DropColumns(cfg).apply(csv_structural)

    assert col not in result.columns
    assert all(c in result.columns for c in csv_structural.columns if c != col)


def test_rename_columns(csv_structural):
    mapping = {"id": "record_id", "name": "full_name"}
    cfg = StepConfig(step="rename_columns", params={"mapping": mapping})
    result, _ = RenameColumns(cfg).apply(csv_structural)

    for old, new in mapping.items():
        assert new in result.columns
        assert old not in result.columns


def test_cast_dtype(csv_structural):
    col = "quantity"
    cfg = StepConfig(step="cast_dtype", params={"casts": {col: "float32"}})
    result, stats = CastDtype(cfg).apply(csv_structural)

    assert result[col].dtype == pl.Float32
    assert col in stats.dtype_changes


def test_cast_dtype_unknown_raises(csv_structural):
    cfg = StepConfig(step="cast_dtype", params={"casts": {"quantity": "matrix"}})
    with pytest.raises(ValueError, match="Unknown dtype"):
        CastDtype(cfg).apply(csv_structural)


def test_deduplicate_all(csv_structural):
    original_len = len(csv_structural)
    expected_len = csv_structural.unique().shape[0]
    duplicates = original_len - expected_len

    cfg = StepConfig(step="deduplicate")
    result, stats = Deduplicate(cfg).apply(csv_structural)

    assert len(result) == expected_len
    assert stats.row_delta == -duplicates


def test_deduplicate_subset(csv_structural):
    col = "name"
    cfg = StepConfig(step="deduplicate", columns=[col])
    result, stats = Deduplicate(cfg).apply(csv_structural)

    assert result[col].n_unique() == len(result)