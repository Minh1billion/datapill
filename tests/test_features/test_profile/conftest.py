from pathlib import Path

import polars as pl
import pytest

from dataprep.storage.artifact import ArtifactStore

_HERE = Path(__file__).resolve().parent
DATA_CSV = _HERE.parents[1] / "fixtures" / "data.csv"


@pytest.fixture(scope="session")
def artifact_store(tmp_path_factory: pytest.TempPathFactory) -> ArtifactStore:
    path = tmp_path_factory.mktemp("artifacts")
    return ArtifactStore(str(path))


@pytest.fixture(scope="session")
def sample_df() -> pl.DataFrame:
    return pl.read_csv(DATA_CSV, try_parse_dates=True)