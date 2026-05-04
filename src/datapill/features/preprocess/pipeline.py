from typing import Any, AsyncGenerator

from ...core.context import Context
from ...core.events import EventType, ProgressEvent
from ...storage.artifact_store import Artifact
from ..base import ExecutionPlan, Pipeline, ValidationResult
from ...utils.loader import load_dataframe
from .ops import apply_op, validate_op

_VALID_PARENTS = {"ingest", "preprocess"}


class PreprocessPipeline(Pipeline):
    def __init__(
        self,
        parent_run_id: str,
        ops: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> None:
        self.parent_run_id = parent_run_id
        self.ops = ops
        self.options = options or {}

    def validate(self, context: Context) -> ValidationResult:
        errors: list[str] = []

        parent = context.artifact_store.get(self.parent_run_id)
        if parent is None:
            errors.append(f"artifact not found: {self.parent_run_id!r}")
            return ValidationResult(ok=False, errors=errors)

        if parent.pipeline not in _VALID_PARENTS:
            errors.append(
                f"preprocess cannot accept input from {parent.pipeline!r}, "
                f"expected one of: {sorted(_VALID_PARENTS)}"
            )

        if not self.ops:
            errors.append("ops list is empty")

        for i, op in enumerate(self.ops):
            op_errors = validate_op(op)
            for e in op_errors:
                errors.append(f"ops[{i}]: {e}")

        return ValidationResult(ok=not errors, errors=errors)

    def plan(self, context: Context) -> ExecutionPlan:
        parent = context.artifact_store.get(self.parent_run_id)
        load_mode = "parquet" if (parent and parent.materialized) else "connector"

        join_ops = [op for op in self.ops if op.get("type") == "join"]
        regular_ops = [op for op in self.ops if op.get("type") != "join"]

        steps: list[dict[str, Any]] = [
            {"action": "load_data", "mode": load_mode, "parent_run_id": self.parent_run_id},
        ]

        for op in join_ops:
            right_parent = context.artifact_store.get(op["right_run_id"])
            steps.append({
                "action": "join",
                "right_run_id": op["right_run_id"],
                "right_load_mode": "parquet" if (right_parent and right_parent.materialized) else "connector",
                "on": op.get("on"),
                "how": op.get("how", "inner"),
            })

        for op in regular_ops:
            steps.append({"action": "apply_op", "op": op})

        if self.options.get("materialized"):
            steps.append({"action": "materialize", "format": "parquet"})

        return ExecutionPlan(
            steps=steps,
            metadata={
                "parent_run_id": self.parent_run_id,
                "load_mode": load_mode,
                "n_ops": len(self.ops),
                "options": self.options,
            },
        )

    async def execute(
        self, plan: ExecutionPlan, context: Context
    ) -> AsyncGenerator[ProgressEvent, None]:
        parent = context.artifact_store.get(self.parent_run_id)
        artifact = Artifact.new(
            pipeline="preprocess",
            parent=parent,
            options={
                "ops": self.ops,
                **self.options,
            },
            is_sample=parent.is_sample if parent else False,
            sample_size=parent.sample_size if parent else None,
        )

        yield ProgressEvent(event_type=EventType.STARTED, message="loading data from parent artifact")

        try:
            df = await load_dataframe(parent, context)
        except Exception as exc:
            yield ProgressEvent(event_type=EventType.ERROR, message=str(exc))
            return

        yield ProgressEvent(
            event_type=EventType.PROGRESS,
            message=f"loaded {len(df):,} rows, {len(df.columns)} columns",
            progress_pct=10.0,
            payload={"rows": len(df), "columns": len(df.columns)},
        )

        join_ops = [op for op in self.ops if op.get("type") == "join"]
        for join_op in join_ops:
            right_artifact = context.artifact_store.get(join_op["right_run_id"])
            if right_artifact is None:
                yield ProgressEvent(
                    event_type=EventType.ERROR,
                    message=f"join: artifact not found: {join_op['right_run_id']!r}",
                )
                return

            yield ProgressEvent(
                event_type=EventType.PROGRESS,
                message=f"loading right side for join: {join_op['right_run_id']}",
                progress_pct=12.0,
            )

            try:
                right_df = await load_dataframe(right_artifact, context)
            except Exception as exc:
                yield ProgressEvent(event_type=EventType.ERROR, message=f"join load failed: {exc}")
                return

            df = df.join(
                right_df,
                on=join_op.get("on"),
                how=join_op.get("how", "inner"),
                suffix=join_op.get("suffix", "_right"),
            )

            yield ProgressEvent(
                event_type=EventType.PROGRESS,
                message=f"join complete — {len(df):,} rows",
                progress_pct=15.0,
                payload={"rows": len(df)},
            )

        regular_ops = [op for op in self.ops if op.get("type") != "join"]
        n_ops = len(regular_ops)

        for i, op in enumerate(regular_ops):
            try:
                df = apply_op(df, op)
            except Exception as exc:
                yield ProgressEvent(
                    event_type=EventType.ERROR,
                    message=f"op {op.get('group')}.{op.get('type')} failed: {exc}",
                )
                return

            pct = 15.0 + ((i + 1) / n_ops) * 75.0
            yield ProgressEvent(
                event_type=EventType.PROGRESS,
                message=f"applied {op.get('group')}.{op.get('type')} ({i + 1}/{n_ops})",
                progress_pct=round(pct, 2),
            )

        artifact.schema = {c: str(df[c].dtype) for c in df.columns}
        artifact.options = {
            **artifact.options,
            "ops": self.ops,
        }

        if self.options.get("materialized"):
            out = context.artifact_store.path / "artifacts" / artifact.run_id / "data.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out)
            artifact.materialized = True
            artifact.path = str(out.relative_to(context.artifact_store.path))
            yield ProgressEvent(
                event_type=EventType.PROGRESS,
                message=f"materialized to {artifact.path}",
                progress_pct=95.0,
                payload={"path": artifact.path},
            )

        context.artifact_store.save(artifact)
        context.artifact = artifact

        fit_params = {
            f"{op.get('group')}.{op.get('type')}[{i}]": op["fit_params"]
            for i, op in enumerate(self.ops)
            if op.get("fit_params")
        }

        yield ProgressEvent(
            event_type=EventType.DONE,
            message="preprocess complete",
            progress_pct=100.0,
            payload={
                "run_id": artifact.run_id,
                "ref": artifact.ref,
                "rows": len(df),
                "columns": len(df.columns),
                **({"fit_params": fit_params} if fit_params else {}),
            },
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "pipeline": "preprocess",
            "version": "1.0",
            "parent_run_id": self.parent_run_id,
            "ops": self.ops,
            "options": self.options,
            "schema": {"input": "parquet | connector", "output": "parquet | none"},
            "capabilities": ["materialize", "join", "fit_params", "stream"],
        }