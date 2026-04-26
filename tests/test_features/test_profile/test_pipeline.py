import asyncio

import polars as pl
import pytest

from dataprep.features.profile.pipeline import ProfileOptions, ProfilePipeline


class _FakeArtifactStore:
    def __init__(self, df: pl.DataFrame, tmp_path):
        self._df = df
        self._store: dict[str, dict] = {}
        self._tmp = tmp_path

    def scan_parquet(self, artifact_id: str):
        path = self._tmp / f"{artifact_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._df.write_parquet(path)
        return pl.scan_parquet(path)

    async def save_json(self, artifact_id: str, data: dict) -> None:
        self._store[artifact_id] = data

    async def load_json(self, artifact_id: str) -> dict:
        return self._store[artifact_id]


class _FakeContext:
    def __init__(self, df: pl.DataFrame, tmp_path):
        self.artifact_store = _FakeArtifactStore(df, tmp_path)


def _collect_events(pipeline, plan, ctx):
    events = []

    async def _run():
        async for e in pipeline.execute(plan, ctx):
            events.append(e)

    asyncio.run(_run())
    return events


def _run_from_df(df, options, tmp_path):
    ctx = _FakeContext(df, tmp_path)
    pipeline = ProfilePipeline(options)
    plan = pipeline.plan(ctx)
    plan.metadata["dataframe"] = df
    events = _collect_events(pipeline, plan, ctx)
    done = next(e for e in events if e.event_type.name == "DONE")
    return done.payload, ctx, events


def _run_from_artifact(df, options, tmp_path):
    ctx = _FakeContext(df, tmp_path)
    pipeline = ProfilePipeline(options)
    plan = pipeline.plan(ctx)
    plan.metadata["input_artifact_id"] = "test_artifact"
    events = _collect_events(pipeline, plan, ctx)
    done = next(e for e in events if e.event_type.name == "DONE")
    return done.payload, ctx, events


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c, d in zip(df.columns, df.dtypes) if d in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)]


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


@pytest.fixture
def small_df():
    return pl.DataFrame({
        "id":     list(range(1, 51)),
        "score":  [round(i * 1.5 + 0.1, 2) for i in range(1, 51)],
        "value":  [float(i * 2) for i in range(1, 51)],
        "name":   ["user_" + str(i) for i in range(1, 51)],
        "active": [i % 2 == 0 for i in range(1, 51)],
    })


@pytest.fixture
def null_df():
    return pl.DataFrame({
        "x": [1.0, None, 3.0, None, 5.0],
        "y": [10.0, 20.0, None, 40.0, 50.0],
        "z": ["a", "b", None, "d", None],
    })


@pytest.fixture
def constant_df():
    return pl.DataFrame({
        "value": [42] * 20,
        "label": ["same"] * 20,
    })


@pytest.fixture
def high_null_df():
    return pl.DataFrame({"sparse": [None] * 90 + [1.0] * 10})


class TestValidation:
    def test_valid_full_mode(self, small_df, tmp):
        ctx = _FakeContext(small_df, tmp)
        result = ProfilePipeline(ProfileOptions(mode="full")).validate(ctx)
        assert result.ok

    def test_valid_summary_mode(self, small_df, tmp):
        ctx = _FakeContext(small_df, tmp)
        result = ProfilePipeline(ProfileOptions(mode="summary")).validate(ctx)
        assert result.ok

    def test_invalid_mode(self, small_df, tmp):
        ctx = _FakeContext(small_df, tmp)
        result = ProfilePipeline(ProfileOptions(mode="invalid")).validate(ctx)
        assert not result.ok
        assert any("mode" in e.lower() for e in result.errors)

    def test_invalid_sample_strategy(self, small_df, tmp):
        ctx = _FakeContext(small_df, tmp)
        result = ProfilePipeline(ProfileOptions(sample_strategy="bad")).validate(ctx)
        assert not result.ok
        assert any("sample_strategy" in e.lower() for e in result.errors)


class TestExecuteFromDataframe:
    def test_done_payload_has_required_keys(self, small_df, tmp):
        payload, _, _ = _run_from_df(small_df, ProfileOptions(), tmp)
        assert "profile_id" in payload
        assert "detail_artifact_id" in payload
        assert "summary_artifact_id" in payload
        assert "duration_s" in payload

    def test_detail_columns_match_dataframe(self, small_df, tmp):
        payload, ctx, _ = _run_from_df(small_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert {c["name"] for c in detail["columns"]} == set(small_df.columns)

    def test_detail_row_count(self, small_df, tmp):
        payload, ctx, _ = _run_from_df(small_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert detail["dataset"]["row_count"] == len(small_df)

    def test_detail_column_count(self, small_df, tmp):
        payload, ctx, _ = _run_from_df(small_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert detail["dataset"]["column_count"] == len(small_df.columns)

    def test_no_input_emits_error_event(self, small_df, tmp):
        ctx = _FakeContext(small_df, tmp)
        pipeline = ProfilePipeline(ProfileOptions())
        plan = pipeline.plan(ctx)
        events = _collect_events(pipeline, plan, ctx)
        error_events = [e for e in events if e.event_type.name == "ERROR"]
        assert len(error_events) == 1
        assert "No input" in error_events[0].message

    def test_with_real_data(self, sample_df, tmp):
        payload, ctx, _ = _run_from_df(sample_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert detail["dataset"]["row_count"] == len(sample_df)
        assert {c["name"] for c in detail["columns"]} == set(sample_df.columns)


class TestExecuteFromArtifact:
    def test_load_from_artifact_id(self, small_df, tmp):
        payload, _, _ = _run_from_artifact(small_df, ProfileOptions(), tmp)
        assert payload["profile_id"] is not None

    def test_schema_hash_present_and_length(self, small_df, tmp):
        payload, ctx, _ = _run_from_artifact(small_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert "schema_hash" in detail["dataset"]
        assert len(detail["dataset"]["schema_hash"]) == 12

    def test_same_schema_same_hash(self, small_df, tmp):
        p1, ctx1, _ = _run_from_artifact(small_df, ProfileOptions(), tmp / "a")
        p2, ctx2, _ = _run_from_artifact(small_df, ProfileOptions(), tmp / "b")
        (tmp / "a").mkdir(exist_ok=True)
        (tmp / "b").mkdir(exist_ok=True)
        d1 = asyncio.run(ctx1.artifact_store.load_json(p1["detail_artifact_id"]))
        d2 = asyncio.run(ctx2.artifact_store.load_json(p2["detail_artifact_id"]))
        assert d1["dataset"]["schema_hash"] == d2["dataset"]["schema_hash"]


class TestSampling:
    def test_no_sampling_returns_full_rows(self, tmp):
        df = pl.DataFrame({"x": list(range(200)), "y": list(range(200))})
        payload, ctx, _ = _run_from_df(df, ProfileOptions(sample_strategy="none"), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert detail["dataset"]["row_count"] == 200
        assert detail["dataset"]["sampled"] is False

    def test_random_sampling_reduces_rows(self, tmp):
        df = pl.DataFrame({"x": list(range(200)), "y": list(range(200))})
        opts = ProfileOptions(sample_strategy="random", sample_size=50)
        payload, ctx, _ = _run_from_df(df, opts, tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert detail["dataset"]["sampled"] is True
        assert detail["dataset"]["row_count"] <= 50

    def test_reservoir_sampling_reduces_rows(self, tmp):
        df = pl.DataFrame({"x": list(range(200)), "y": list(range(200))})
        opts = ProfileOptions(sample_strategy="reservoir", sample_size=50)
        payload, ctx, _ = _run_from_df(df, opts, tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert detail["dataset"]["sampled"] is True
        assert detail["dataset"]["row_count"] <= 50

    def test_sampling_skipped_when_df_smaller_than_sample_size(self, tmp):
        df = pl.DataFrame({"x": list(range(30)), "y": list(range(30))})
        opts = ProfileOptions(sample_strategy="random", sample_size=100)
        payload, ctx, _ = _run_from_df(df, opts, tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert detail["dataset"]["row_count"] == 30


class TestColumnStats:
    def test_numeric_column_has_mean(self, small_df, tmp):
        payload, ctx, _ = _run_from_df(small_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        score_col = next(c for c in detail["columns"] if c["name"] == "score")
        assert score_col["mean"] is not None

    def test_string_column_has_top_values(self, small_df, tmp):
        payload, ctx, _ = _run_from_df(small_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        name_col = next(c for c in detail["columns"] if c["name"] == "name")
        assert name_col["top_values"] is not None

    def test_null_pct_computed_correctly(self, null_df, tmp):
        payload, ctx, _ = _run_from_df(null_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        x_col = next(c for c in detail["columns"] if c["name"] == "x")
        assert abs(x_col["null_pct"] - 0.4) < 0.01

    def test_constant_column_warning(self, constant_df, tmp):
        payload, ctx, _ = _run_from_df(constant_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        value_col = next(c for c in detail["columns"] if c["name"] == "value")
        assert any("CONSTANT" in w for w in value_col["warnings"])

    def test_high_null_warning(self, high_null_df, tmp):
        payload, ctx, _ = _run_from_df(high_null_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert any("HIGH_NULL" in w for w in detail["columns"][0]["warnings"])

    def test_real_data_all_columns_profiled(self, sample_df, tmp):
        payload, ctx, _ = _run_from_df(sample_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert len(detail["columns"]) == len(sample_df.columns)


class TestCorrelation:
    def _has_enough_numeric(self, df: pl.DataFrame) -> bool:
        return len(_numeric_cols(df)) >= 2

    def test_correlation_none_skips(self, small_df, tmp):
        opts = ProfileOptions(correlation_method="none")
        payload, ctx, _ = _run_from_df(small_df, opts, tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert detail["correlations"] == []

    def test_pearson_returns_list(self, small_df, tmp):
        opts = ProfileOptions(correlation_method="pearson", correlation_threshold=0.0)
        payload, ctx, _ = _run_from_df(small_df, opts, tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert isinstance(detail["correlations"], list)

    def test_pearson_with_correlated_data(self, tmp):
        n = 50
        x = list(range(n))
        df = pl.DataFrame({
            "a": [float(i) for i in x],
            "b": [float(i) * 2 + 1 for i in x],
            "c": [float(i) * -1 for i in x],
        })
        opts = ProfileOptions(correlation_method="pearson", correlation_threshold=0.0)
        payload, ctx, _ = _run_from_df(df, opts, tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert len(detail["correlations"]) > 0

    def test_spearman_with_correlated_data(self, tmp):
        n = 50
        x = list(range(n))
        df = pl.DataFrame({
            "a": [float(i) for i in x],
            "b": [float(i) * 2 + 1 for i in x],
            "c": [float(i) * -1 for i in x],
        })
        opts = ProfileOptions(correlation_method="spearman", correlation_threshold=0.0)
        payload, ctx, _ = _run_from_df(df, opts, tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert len(detail["correlations"]) > 0

    def test_threshold_filters_pairs(self, tmp):
        n = 50
        x = list(range(n))
        df = pl.DataFrame({
            "a": [float(i) for i in x],
            "b": [float(i) * 2 + 1 for i in x],
            "c": [float(i) * -1 for i in x],
        })
        opts_all = ProfileOptions(correlation_method="pearson", correlation_threshold=0.0)
        opts_high = ProfileOptions(correlation_method="pearson", correlation_threshold=0.99)
        p_all, ctx_all, _ = _run_from_df(df, opts_all, tmp / "all")
        p_high, ctx_high, _ = _run_from_df(df, opts_high, tmp / "high")
        (tmp / "all").mkdir(exist_ok=True)
        (tmp / "high").mkdir(exist_ok=True)
        d_all = asyncio.run(ctx_all.artifact_store.load_json(p_all["detail_artifact_id"]))
        d_high = asyncio.run(ctx_high.artifact_store.load_json(p_high["detail_artifact_id"]))
        assert len(d_all["correlations"]) >= len(d_high["correlations"])

    def test_real_data_pearson(self, sample_df, tmp):
        if not self._has_enough_numeric(sample_df):
            pytest.skip("sample_df has fewer than 2 numeric columns")
        opts = ProfileOptions(correlation_method="pearson", correlation_threshold=0.0)
        payload, ctx, _ = _run_from_df(sample_df, opts, tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        assert isinstance(detail["correlations"], list)


class TestSummaryOutput:
    def test_columns_match_detail(self, small_df, tmp):
        payload, ctx, _ = _run_from_df(small_df, ProfileOptions(), tmp)
        detail = asyncio.run(ctx.artifact_store.load_json(payload["detail_artifact_id"]))
        summary = asyncio.run(ctx.artifact_store.load_json(payload["summary_artifact_id"]))
        assert {c["name"] for c in summary["columns"]} == {c["name"] for c in detail["columns"]}

    def test_column_has_required_fields(self, small_df, tmp):
        payload, ctx, _ = _run_from_df(small_df, ProfileOptions(), tmp)
        summary = asyncio.run(ctx.artifact_store.load_json(payload["summary_artifact_id"]))
        for col in summary["columns"]:
            assert "name" in col
            assert "dtype" in col
            assert "null_pct" in col
            assert "distinct_count" in col

    def test_dataset_fields(self, small_df, tmp):
        payload, ctx, _ = _run_from_df(small_df, ProfileOptions(), tmp)
        summary = asyncio.run(ctx.artifact_store.load_json(payload["summary_artifact_id"]))
        assert summary["dataset"]["row_count"] == len(small_df)
        assert summary["dataset"]["column_count"] == len(small_df.columns)

    def test_high_level_warnings_deduplicated(self, constant_df, tmp):
        payload, ctx, _ = _run_from_df(constant_df, ProfileOptions(), tmp)
        summary = asyncio.run(ctx.artifact_store.load_json(payload["summary_artifact_id"]))
        types_seen = [w.split(":")[0].strip() for w in summary["high_level_warnings"]]
        assert len(types_seen) == len(set(types_seen))


class TestProgressEvents:
    def test_first_event_is_started(self, small_df, tmp):
        _, _, events = _run_from_df(small_df, ProfileOptions(), tmp)
        assert events[0].event_type.name == "STARTED"

    def test_last_event_is_done(self, small_df, tmp):
        _, _, events = _run_from_df(small_df, ProfileOptions(), tmp)
        assert events[-1].event_type.name == "DONE"

    def test_done_progress_pct_is_one(self, small_df, tmp):
        _, _, events = _run_from_df(small_df, ProfileOptions(), tmp)
        assert events[-1].progress_pct == 1.0

    def test_progress_pct_monotonically_increases(self, small_df, tmp):
        _, _, events = _run_from_df(small_df, ProfileOptions(), tmp)
        pcts = [e.progress_pct for e in events if e.progress_pct is not None]
        assert pcts == sorted(pcts)


class TestSerialize:
    def test_returns_feature_key(self):
        s = ProfilePipeline(ProfileOptions()).serialize()
        assert s["feature"] == "profile"

    def test_options_reflected(self):
        s = ProfilePipeline(ProfileOptions(mode="summary", correlation_method="spearman")).serialize()
        assert s["options"]["mode"] == "summary"
        assert s["options"]["correlation_method"] == "spearman"