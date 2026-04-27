import pytest
import polars as pl

from dataprep.features.export.writers import read, write


def _roundtrip(df, tmp_path, fmt, filename, **opts):
    path = tmp_path / filename
    write(df, path, fmt, **opts)
    assert path.exists()
    result = read(path, fmt)
    assert set(result.columns) == set(df.columns)
    assert len(result) == len(df)
    return result


def test_csv_roundtrip(sample_df, tmp_path):
    result = _roundtrip(sample_df, tmp_path, "csv", "out.csv")
    assert result["id"].to_list() == sample_df["id"].to_list()


def test_csv_custom_delimiter(sample_df, tmp_path):
    path = tmp_path / "out.csv"
    write(sample_df, path, "csv", delimiter=";")
    content = path.read_text()
    assert ";" in content


def test_parquet_roundtrip_snappy(sample_df, tmp_path):
    result = _roundtrip(sample_df, tmp_path, "parquet", "out.parquet", compression="snappy")
    assert result["score"].to_list() == pytest.approx(sample_df["score"].to_list())


def test_parquet_roundtrip_zstd(sample_df, tmp_path):
    _roundtrip(sample_df, tmp_path, "parquet", "out_zstd.parquet", compression="zstd")


def test_parquet_roundtrip_gzip(sample_df, tmp_path):
    _roundtrip(sample_df, tmp_path, "parquet", "out_gzip.parquet", compression="gzip")


def test_excel_roundtrip(sample_df, tmp_path):
    result = _roundtrip(sample_df, tmp_path, "excel", "out.xlsx", sheet_name="Data")
    assert len(result) == len(sample_df)


def test_excel_freeze_header(sample_df, tmp_path):
    path = tmp_path / "out.xlsx"
    write(sample_df, path, "excel", freeze_header=True)
    assert path.exists()


def test_json_roundtrip(sample_df, tmp_path):
    result = _roundtrip(sample_df, tmp_path, "json", "out.json")
    assert set(result.columns) == set(sample_df.columns)


def test_jsonl_roundtrip(sample_df, tmp_path):
    result = _roundtrip(sample_df, tmp_path, "jsonl", "out.jsonl", jsonl=True)
    assert len(result) == len(sample_df)


def test_arrow_roundtrip(sample_df, tmp_path):
    pytest.importorskip("pyarrow")
    result = _roundtrip(sample_df, tmp_path, "arrow", "out.feather")
    assert result["id"].to_list() == sample_df["id"].to_list()


def test_unsupported_format_raises(sample_df, tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        write(sample_df, tmp_path / "out.avro", "avro")


def test_unsupported_read_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format for read"):
        read(tmp_path / "out.avro", "avro")