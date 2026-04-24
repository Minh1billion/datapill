import pytest
import polars as pl
from pathlib import Path

from dataprep.storage.artifact import ArtifactStore


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(base_path=str(tmp_path / "artifacts"))


@pytest.fixture
def simple_df() -> pl.DataFrame:
    return pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [1.1, 2.2, 3.3]})


class TestArtifactStoreJson:
    @pytest.mark.asyncio
    async def test_save_and_load_json(self, store):
        data = {"key": "value", "count": 42, "items": [1, 2, 3]}
        await store.save_json("test_json", data)
        loaded = await store.load_json("test_json")
        assert loaded == data

    @pytest.mark.asyncio
    async def test_save_json_returns_path(self, store):
        path = await store.save_json("test_path", {"x": 1})
        assert path.exists()
        assert path.suffix == ".json"

    @pytest.mark.asyncio
    async def test_save_json_nested_dict(self, store):
        data = {"nested": {"a": 1, "b": [2, 3]}, "list": [{"x": 1}]}
        await store.save_json("nested", data)
        loaded = await store.load_json("nested")
        assert loaded == data

    @pytest.mark.asyncio
    async def test_save_json_with_non_serializable_uses_str(self, store):
        import datetime
        data = {"ts": datetime.datetime(2024, 1, 1)}
        path = await store.save_json("ts_test", data)
        assert path.exists()

    @pytest.mark.asyncio
    async def test_load_missing_json_raises(self, store):
        with pytest.raises(FileNotFoundError):
            await store.load_json("nonexistent")

    @pytest.mark.asyncio
    async def test_overwrite_json(self, store):
        await store.save_json("overwrite", {"v": 1})
        await store.save_json("overwrite", {"v": 2})
        loaded = await store.load_json("overwrite")
        assert loaded["v"] == 2


class TestArtifactStoreParquet:
    @pytest.mark.asyncio
    async def test_save_and_load_parquet(self, store, simple_df):
        await store.save_parquet("test_pq", simple_df)
        loaded = store.load_parquet("test_pq")
        assert loaded.shape == simple_df.shape
        assert loaded.columns == simple_df.columns

    @pytest.mark.asyncio
    async def test_save_parquet_returns_path(self, store, simple_df):
        path = await store.save_parquet("pq_path", simple_df)
        assert path.exists()
        assert path.suffix == ".parquet"

    def test_scan_parquet_lazy(self, store, simple_df):
        import asyncio
        asyncio.get_event_loop().run_until_complete(store.save_parquet("scan_test", simple_df))
        lf = store.scan_parquet("scan_test")
        assert isinstance(lf, pl.LazyFrame)
        assert lf.collect().shape == simple_df.shape

    @pytest.mark.asyncio
    async def test_load_missing_parquet_raises(self, store):
        with pytest.raises(Exception):
            store.load_parquet("nonexistent")

    @pytest.mark.asyncio
    async def test_parquet_roundtrip_preserves_dtypes(self, store):
        df = pl.DataFrame({
            "int_col": pl.Series([1, 2], dtype=pl.Int32),
            "float_col": pl.Series([1.0, 2.0], dtype=pl.Float64),
            "str_col": ["a", "b"],
            "bool_col": [True, False],
        })
        await store.save_parquet("dtype_test", df)
        loaded = store.load_parquet("dtype_test")
        assert loaded["int_col"].dtype == pl.Int32
        assert loaded["float_col"].dtype == pl.Float64
        assert loaded["bool_col"].dtype == pl.Boolean


class TestArtifactStoreExists:
    @pytest.mark.asyncio
    async def test_exists_true_after_save(self, store):
        await store.save_json("exists_test", {"x": 1})
        assert store.exists("exists_test", "json") is True

    def test_exists_false_for_missing(self, store):
        assert store.exists("missing_artifact", "json") is False

    @pytest.mark.asyncio
    async def test_exists_checks_extension(self, store, simple_df):
        await store.save_parquet("ext_test", simple_df)
        assert store.exists("ext_test", "parquet") is True
        assert store.exists("ext_test", "json") is False


class TestArtifactStoreChecksum:
    @pytest.mark.asyncio
    async def test_checksum_consistent(self, store):
        await store.save_json("chk", {"x": 1})
        h1 = store.checksum("chk", "json")
        h2 = store.checksum("chk", "json")
        assert h1 == h2

    @pytest.mark.asyncio
    async def test_checksum_changes_on_update(self, store):
        await store.save_json("chk2", {"x": 1})
        h1 = store.checksum("chk2", "json")
        await store.save_json("chk2", {"x": 2})
        h2 = store.checksum("chk2", "json")
        assert h1 != h2

    @pytest.mark.asyncio
    async def test_checksum_is_hex_string(self, store):
        await store.save_json("hex_test", {"y": 99})
        h = store.checksum("hex_test", "json")
        assert isinstance(h, str)
        int(h, 16)


class TestArtifactStoreInit:
    def test_creates_directory_if_not_exists(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "artifacts"
        store = ArtifactStore(base_path=str(path))
        assert path.exists()

    def test_default_base_path_creates(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = ArtifactStore()
        assert store.base.exists()