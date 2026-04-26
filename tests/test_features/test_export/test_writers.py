from pathlib import Path

import polars as pl
import pytest

from dataprep.features.export.writers import read, write


@pytest.fixture
def df():
    return pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "carol"],
        "score": [1.1, 2.2, 3.3],
        "active": [True, False, True],
    })


def _roundtrip(df, tmp_path, fmt, filename, **opts):
    path = tmp_path / filename
    write(df, path, fmt, **opts)
    assert path.exists()
    result = read(path, fmt)
    assert result.columns == df.columns
    assert len(result) == len(df)
    return result


def test_csv_roundtrip(df, tmp_path):
    result = _roundtrip(df, tmp_path, "csv", "out.csv")
    assert result["id"].to_list() == df["id"].to_list()


def test_csv_custom_delimiter(df, tmp_path):
    path = tmp_path / "out.csv"
    write(df, path, "csv", delimiter=";")
    content = path.read_text()
    assert ";" in content


def test_parquet_roundtrip_snappy(df, tmp_path):
    result = _roundtrip(df, tmp_path, "parquet", "out.parquet", compression="snappy")
    assert result["score"].to_list() == pytest.approx(df["score"].to_list())


def test_parquet_roundtrip_zstd(df, tmp_path):
    _roundtrip(df, tmp_path, "parquet", "out_zstd.parquet", compression="zstd")


def test_parquet_roundtrip_gzip(df, tmp_path):
    _roundtrip(df, tmp_path, "parquet", "out_gzip.parquet", compression="gzip")


def test_excel_roundtrip(df, tmp_path):
    result = _roundtrip(df, tmp_path, "excel", "out.xlsx", sheet_name="Data")
    assert len(result) == len(df)


def test_excel_freeze_header(df, tmp_path):
    path = tmp_path / "out.xlsx"
    write(df, path, "excel", freeze_header=True)
    assert path.exists()


def test_json_roundtrip(df, tmp_path):
    result = _roundtrip(df, tmp_path, "json", "out.json")
    assert set(result.columns) == set(df.columns)


def test_jsonl_roundtrip(df, tmp_path):
    result = _roundtrip(df, tmp_path, "jsonl", "out.jsonl", jsonl=True)
    assert len(result) == len(df)


def test_arrow_roundtrip(df, tmp_path):
    pytest.importorskip("pyarrow")
    result = _roundtrip(df, tmp_path, "arrow", "out.feather")
    assert result["id"].to_list() == df["id"].to_list()


def test_unsupported_format_raises(df, tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        write(df, tmp_path / "out.avro", "avro")


def test_unsupported_read_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format for read"):
        read(tmp_path / "out.avro", "avro")