import pytest
import polars as pl
from pathlib import Path

from dataprep.features.profile.pipeline import ProfilePipeline, ProfileOptions, _schema_hash, _reservoir_sample
from dataprep.core.events import EventType
from dataprep.core.interfaces import ExecutionPlan

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestProfileOptions:
    def test_defaults(self):
        opts = ProfileOptions()
        assert opts.mode == "full"
        assert opts.sample_strategy == "none"
        assert opts.sample_size == 100_000
        assert opts.correlation_method == "pearson"
        assert opts.correlation_threshold == 0.5


class TestSchemaHash:
    def test_same_schema_same_hash(self, sample_df):
        h1 = _schema_hash(sample_df)
        h2 = _schema_hash(sample_df)
        assert h1 == h2

    def test_different_schema_different_hash(self, sample_df):
        df2 = sample_df.rename({sample_df.columns[0]: "___renamed___"})
        assert _schema_hash(sample_df) != _schema_hash(df2)

    def test_hash_length(self, sample_df):
        assert len(_schema_hash(sample_df)) == 12


class TestReservoirSample:
    def test_sample_smaller_than_k_returns_all(self, sample_df):
        k = len(sample_df) + 100
        result = _reservoir_sample(sample_df, k)
        assert len(result) == len(sample_df)

    def test_sample_returns_k_rows(self, sample_df):
        k = max(1, len(sample_df) // 2)
        result = _reservoir_sample(sample_df, k)
        assert len(result) == k

    def test_sample_rows_are_from_original(self, sample_df):
        k = max(1, len(sample_df) // 2)
        result = _reservoir_sample(sample_df, k)
        assert set(result.columns) == set(sample_df.columns)


class TestProfilePipelineValidation:
    def test_valid_options_passes(self, pipeline_context):
        pipeline = ProfilePipeline(ProfileOptions(mode="full", sample_strategy="none"))
        result = pipeline.validate(pipeline_context)
        assert result.ok is True

    def test_invalid_mode_fails(self, pipeline_context):
        pipeline = ProfilePipeline(ProfileOptions(mode="invalid"))
        result = pipeline.validate(pipeline_context)
        assert result.ok is False
        assert any("mode" in e for e in result.errors)

    def test_invalid_sample_strategy_fails(self, pipeline_context):
        pipeline = ProfilePipeline(ProfileOptions(sample_strategy="bad"))
        result = pipeline.validate(pipeline_context)
        assert result.ok is False
        assert any("sample_strategy" in e for e in result.errors)

    def test_summary_mode_valid(self, pipeline_context):
        pipeline = ProfilePipeline(ProfileOptions(mode="summary"))
        result = pipeline.validate(pipeline_context)
        assert result.ok is True


class TestProfilePipelinePlan:
    def test_plan_has_required_steps(self, pipeline_context):
        pipeline = ProfilePipeline()
        plan = pipeline.plan(pipeline_context)
        names = [s["name"] for s in plan.steps]
        assert "load_data" in names
        assert "compute_stats" in names
        assert "build_output" in names


class TestProfilePipelineExecute:
    @pytest.mark.asyncio
    async def test_execute_emits_started(self, sample_df, pipeline_context):
        pipeline = ProfilePipeline()
        plan = ExecutionPlan(steps=[], metadata={"dataframe": sample_df})
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        assert events[0].event_type == EventType.STARTED

    @pytest.mark.asyncio
    async def test_execute_emits_done(self, sample_df, pipeline_context):
        pipeline = ProfilePipeline()
        plan = ExecutionPlan(steps=[], metadata={"dataframe": sample_df})
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        assert events[-1].event_type == EventType.DONE

    @pytest.mark.asyncio
    async def test_execute_done_payload_has_profile_id(self, sample_df, pipeline_context):
        pipeline = ProfilePipeline()
        plan = ExecutionPlan(steps=[], metadata={"dataframe": sample_df})
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        payload = events[-1].payload
        assert "profile_id" in payload
        assert "detail_artifact_id" in payload
        assert "summary_artifact_id" in payload

    @pytest.mark.asyncio
    async def test_execute_saves_detail_artifact(self, sample_df, pipeline_context):
        pipeline = ProfilePipeline()
        plan = ExecutionPlan(steps=[], metadata={"dataframe": sample_df})
        async for _ in pipeline.execute(plan, pipeline_context):
            pass
        artifacts = [
            f for f in pipeline_context.artifact_store.base.iterdir()
            if "_detail.json" in f.name
        ]
        assert len(artifacts) >= 1

    @pytest.mark.asyncio
    async def test_detail_has_correct_column_count(self, sample_df, pipeline_context):
        pipeline = ProfilePipeline()
        plan = ExecutionPlan(steps=[], metadata={"dataframe": sample_df})
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        detail_id = events[-1].payload["detail_artifact_id"]
        detail = await pipeline_context.artifact_store.load_json(detail_id)
        assert detail["dataset"]["column_count"] == len(sample_df.columns)
        assert detail["dataset"]["row_count"] == len(sample_df)

    @pytest.mark.asyncio
    async def test_detail_column_profiles_complete(self, sample_df, pipeline_context):
        pipeline = ProfilePipeline()
        plan = ExecutionPlan(steps=[], metadata={"dataframe": sample_df})
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        detail_id = events[-1].payload["detail_artifact_id"]
        detail = await pipeline_context.artifact_store.load_json(detail_id)
        assert len(detail["columns"]) == len(sample_df.columns)
        col_names = {c["name"] for c in detail["columns"]}
        assert col_names == set(sample_df.columns)

    @pytest.mark.asyncio
    async def test_summary_has_high_level_structure(self, sample_df, pipeline_context):
        pipeline = ProfilePipeline()
        plan = ExecutionPlan(steps=[], metadata={"dataframe": sample_df})
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        summary_id = events[-1].payload["summary_artifact_id"]
        summary = await pipeline_context.artifact_store.load_json(summary_id)
        assert "dataset" in summary
        assert "columns" in summary
        assert "high_level_warnings" in summary

    @pytest.mark.asyncio
    async def test_progress_pct_ends_at_1(self, sample_df, pipeline_context):
        pipeline = ProfilePipeline()
        plan = ExecutionPlan(steps=[], metadata={"dataframe": sample_df})
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        assert events[-1].progress_pct == 1.0

    @pytest.mark.asyncio
    async def test_no_input_emits_error(self, pipeline_context):
        pipeline = ProfilePipeline()
        plan = ExecutionPlan(steps=[], metadata={})
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        assert any(e.event_type == EventType.ERROR for e in events)

    @pytest.mark.asyncio
    async def test_with_random_sampling(self, sample_df, pipeline_context):
        opts = ProfileOptions(sample_strategy="random", sample_size=max(1, len(sample_df) // 2))
        pipeline = ProfilePipeline(opts)
        plan = ExecutionPlan(steps=[], metadata={"dataframe": sample_df})
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        assert events[-1].event_type == EventType.DONE

    @pytest.mark.asyncio
    async def test_correlation_none_skips(self, sample_df, pipeline_context):
        opts = ProfileOptions(correlation_method="none")
        pipeline = ProfilePipeline(opts)
        plan = ExecutionPlan(steps=[], metadata={"dataframe": sample_df})
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        detail_id = events[-1].payload["detail_artifact_id"]
        detail = await pipeline_context.artifact_store.load_json(detail_id)
        assert detail["correlations"] == []

    @pytest.mark.asyncio
    async def test_execute_with_parquet_artifact(self, sample_df, pipeline_context):
        await pipeline_context.artifact_store.save_parquet("input_data", sample_df)
        pipeline = ProfilePipeline()
        plan = ExecutionPlan(steps=[], metadata={"input_artifact_id": "input_data"})
        events = []
        async for e in pipeline.execute(plan, pipeline_context):
            events.append(e)
        assert events[-1].event_type == EventType.DONE


class TestProfilePipelineSerialize:
    def test_serialize_structure(self):
        pipeline = ProfilePipeline()
        data = pipeline.serialize()
        assert data["feature"] == "profile"
        assert "version" in data
        assert "options" in data

    def test_serialize_options_present(self):
        opts = ProfileOptions(mode="summary", sample_strategy="random", correlation_method="none")
        pipeline = ProfilePipeline(opts)
        data = pipeline.serialize()
        assert data["options"]["mode"] == "summary"
        assert data["options"]["sample_strategy"] == "random"
        assert data["options"]["correlation_method"] == "none"