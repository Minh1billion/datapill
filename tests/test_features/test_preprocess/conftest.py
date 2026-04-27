from collections.abc import Generator
from pathlib import Path

import polars as pl
import pytest

_HERE = Path(__file__).resolve().parent
DATA_CSV = _HERE.parents[1] / "fixtures" / "data.csv"

_NULL_INDICES = [1, 3]


def _write(tmp_path_factory: pytest.TempPathFactory, name: str, df: pl.DataFrame) -> Path:
    p = tmp_path_factory.getbasetemp() / name
    df.write_csv(p)
    return p


@pytest.fixture(scope="session")
def base_df() -> pl.DataFrame:
    return pl.read_csv(DATA_CSV, try_parse_dates=True)


@pytest.fixture(scope="session")
def csv_with_nulls(tmp_path_factory: pytest.TempPathFactory, base_df: pl.DataFrame) -> Generator[pl.DataFrame, None, None]:
    mask = pl.Series("mask", [i in _NULL_INDICES for i in range(len(base_df))])
    df = base_df.with_columns([
        pl.when(mask).then(None).otherwise(pl.col("score")).alias("score"),
        pl.when(mask).then(None).otherwise(pl.col("category")).alias("category"),
    ])
    p = _write(tmp_path_factory, "data_nulls.csv", df)
    yield pl.read_csv(p, try_parse_dates=True)
    p.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def csv_with_outliers(tmp_path_factory: pytest.TempPathFactory, base_df: pl.DataFrame) -> Generator[pl.DataFrame, None, None]:
    q3 = base_df["value"].quantile(0.75)
    iqr = q3 - base_df["value"].quantile(0.25)
    outlier_val = q3 + 10 * iqr
    extra = base_df.head(1).with_columns(pl.lit(outlier_val).alias("value"))
    df = pl.concat([base_df, extra])
    p = _write(tmp_path_factory, "data_outliers.csv", df)
    yield pl.read_csv(p, try_parse_dates=True)
    p.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def csv_encoding(tmp_path_factory: pytest.TempPathFactory, base_df: pl.DataFrame) -> Generator[pl.DataFrame, None, None]:
    p = _write(tmp_path_factory, "data_encoding.csv", base_df)
    yield pl.read_csv(p, try_parse_dates=True)
    p.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def csv_scaling(tmp_path_factory: pytest.TempPathFactory, base_df: pl.DataFrame) -> Generator[pl.DataFrame, None, None]:
    p = _write(tmp_path_factory, "data_scaling.csv", base_df)
    yield pl.read_csv(p, try_parse_dates=True)
    p.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def csv_structural(tmp_path_factory: pytest.TempPathFactory, base_df: pl.DataFrame) -> Generator[pl.DataFrame, None, None]:
    df = pl.concat([base_df, base_df.head(2)])
    p = _write(tmp_path_factory, "data_structural.csv", df)
    yield pl.read_csv(p, try_parse_dates=True)
    p.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def csv_pipeline(tmp_path_factory: pytest.TempPathFactory, base_df: pl.DataFrame) -> Generator[pl.DataFrame, None, None]:
    q3 = base_df["score"].quantile(0.75)
    iqr = q3 - base_df["score"].quantile(0.25)
    outlier_score = q3 + 10 * iqr
    mask = pl.Series("mask", [i in _NULL_INDICES for i in range(len(base_df))])
    df = base_df.with_columns([
        pl.when(mask).then(None).otherwise(pl.col("score")).alias("score"),
        pl.when(mask).then(None).otherwise(pl.col("category")).alias("category"),
    ])
    extra = df.head(1).with_columns(pl.lit(outlier_score).alias("score"))
    df = pl.concat([df, extra])
    p = _write(tmp_path_factory, "data_pipeline.csv", df)
    yield pl.read_csv(p, try_parse_dates=True)
    p.unlink(missing_ok=True)