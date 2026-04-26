import textwrap

import polars as pl
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dataprep.executor.validator import analyze_ast, validate_schema
from dataprep.executor.sandbox import (
    SandboxConfig,
    run_sample,
    apply,
    run_sql,
    SAMPLE_ROWS,
)


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame({
        "id": list(range(1, 11)),
        "value": [float(i) for i in range(10)],
        "category": ["a", "b"] * 5,
    })


@pytest.fixture
def large_df() -> pl.DataFrame:
    n = SAMPLE_ROWS + 500
    return pl.DataFrame({
        "x": list(range(n)),
        "y": [float(i) * 0.1 for i in range(n)],
    })


def _code(body: str) -> str:
    return textwrap.dedent(f"""\
        def transform(df):
            {textwrap.indent(body.strip(), '    ')}
    """)


class TestASTAnalysis:

    def test_clean_code_passes(self):
        code = _code("return df.with_columns(pl.col('value') * 2)")
        result = analyze_ast(code)
        assert result.ok, result.violations

    def test_import_os_rejected(self):
        code = "import os\n" + _code("return df")
        result = analyze_ast(code)
        assert not result.ok
        assert any("os" in str(v) for v in result.violations)

    def test_import_subprocess_rejected(self):
        code = "import subprocess\n" + _code("return df")
        result = analyze_ast(code)
        assert not result.ok
        assert any("subprocess" in str(v) for v in result.violations)

    def test_import_from_os_rejected(self):
        code = "from os import path\n" + _code("return df")
        result = analyze_ast(code)
        assert not result.ok

    def test_import_socket_rejected(self):
        code = "import socket\n" + _code("return df")
        result = analyze_ast(code)
        assert not result.ok
        assert any("socket" in str(v) for v in result.violations)

    def test_import_requests_rejected(self):
        code = "import requests\n" + _code("return df")
        result = analyze_ast(code)
        assert not result.ok

    def test_import_sys_rejected(self):
        code = "import sys\n" + _code("return df")
        result = analyze_ast(code)
        assert not result.ok

    def test_exec_call_rejected(self):
        code = _code("exec('x=1'); return df")
        result = analyze_ast(code)
        assert not result.ok
        assert any("exec" in str(v) for v in result.violations)

    def test_eval_call_rejected(self):
        code = _code("eval('1+1'); return df")
        result = analyze_ast(code)
        assert not result.ok

    def test_open_call_rejected(self):
        code = _code("open('/etc/passwd'); return df")
        result = analyze_ast(code)
        assert not result.ok
        assert any("open" in str(v) for v in result.violations)

    def test_dunder_attribute_rejected(self):
        code = _code("df.__class__; return df")
        result = analyze_ast(code)
        assert not result.ok
        assert any("__class__" in str(v) for v in result.violations)

    def test_global_statement_rejected(self):
        code = "x = 1\n" + _code("global x; return df")
        result = analyze_ast(code)
        assert not result.ok

    def test_multiple_violations_all_reported(self):
        code = "import os\nimport socket\n" + _code("return df")
        result = analyze_ast(code)
        assert not result.ok
        assert len(result.violations) >= 2

    def test_polars_import_allowed(self):
        """Polars is injected into globals — importing it in code is also fine."""
        code = "import polars as pl\n" + _code("return df.with_columns(pl.col('value') + 1)")
        result = analyze_ast(code)
        assert result.ok, result.violations

    def test_syntax_error_raises(self):
        with pytest.raises(SyntaxError):
            analyze_ast("def transform(df:\n    return df")


class TestSandboxIsolation:
    def test_file_read_blocked(self, df):
        code = _code("open('/etc/passwd', 'r').read(); return df")
        result = run_sample(code, df)
        assert not result.ok

    def test_file_write_blocked(self, df):
        code = _code("open('/tmp/pwned.txt', 'w').write('x'); return df")
        result = run_sample(code, df)
        assert not result.ok

    def test_socket_blocked(self, df):
        code = "import socket\n" + _code("return df")
        result = run_sample(code, df)
        assert not result.ok

    def test_urllib_blocked(self, df):
        code = "import urllib.request\n" + _code("return df")
        result = run_sample(code, df)
        assert not result.ok

    def test_subprocess_blocked(self, df):
        code = "import subprocess\n" + _code("return df")
        result = run_sample(code, df)
        assert not result.ok

    def test_os_system_blocked(self, df):
        code = "import os\n" + _code("os.system('id'); return df")
        result = run_sample(code, df)
        assert not result.ok

    def test_valid_transform_passes(self, df):
        code = _code("return df.with_columns(pl.col('value') * 2)")
        result = run_sample(code, df)
        assert result.ok, result.error_message

    def test_preview_capped_at_100(self, large_df):
        code = _code("return df")
        result = run_sample(code, large_df)
        assert result.ok
        assert result.preview is not None
        assert len(result.preview) <= 100

    def test_sample_uses_at_most_sample_rows(self, large_df):
        code = _code("return df")
        result = run_sample(code, large_df)
        assert result.ok
        assert result.schema_diff["row_count_in"] == SAMPLE_ROWS


class TestRollback:

    def test_apply_raises_on_failure(self, df):
        # Code passes AST check but fails at runtime
        code = _code("raise RuntimeError('intentional failure'); return df")
        with pytest.raises(RuntimeError, match="Full execution failed"):
            apply(code, df)

    def test_original_df_unchanged_after_failure(self, df):
        original_values = df["value"].to_list()
        code = _code("raise RuntimeError('boom'); return df")
        try:
            apply(code, df)
        except RuntimeError:
            pass
        assert df["value"].to_list() == original_values

    def test_apply_success_returns_transformed(self, df):
        code = _code("return df.with_columns((pl.col('value') + 100).alias('value'))")
        result = apply(code, df)
        assert result["value"][0] == 100.0

    def test_type_error_on_wrong_return_type(self, df):
        code = _code("return 42")
        result = run_sample(code, df)
        assert not result.ok
        assert result.failed_step == 2

    def test_missing_transform_function_fails(self, df):
        code = textwrap.dedent("""\
            # No transform function defined
            x = 1 + 1
        """)
        result = run_sample(code, df)
        assert not result.ok
        assert result.failed_step == 2


class TestSQLCustomStep:

    def test_valid_select(self, df):
        result = run_sql("SELECT id, value FROM df WHERE value > 4", df)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5

    def test_select_all(self, df):
        result = run_sql("SELECT * FROM df", df)
        assert len(result) == len(df)

    def test_select_with_expression(self, df):
        result = run_sql("SELECT id, value * 2 AS double_value FROM df", df)
        assert "double_value" in result.columns

    def test_cte_allowed(self, df):
        result = run_sql(
            "WITH cte AS (SELECT * FROM df WHERE value > 3) SELECT * FROM cte",
            df,
        )
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 6

    def test_insert_rejected(self, df):
        with pytest.raises(ValueError, match="INSERT"):
            run_sql("INSERT INTO df VALUES (99, 9.9, 'z')", df)

    def test_update_rejected(self, df):
        with pytest.raises(ValueError, match="Only SELECT"):
            run_sql("UPDATE df SET value = 0", df)

    def test_delete_rejected(self, df):
        with pytest.raises(ValueError, match="Only SELECT"):
            run_sql("DELETE FROM df WHERE id = 1", df)

    def test_drop_rejected(self, df):
        with pytest.raises(ValueError, match="DROP"):
            run_sql("SELECT * FROM df; DROP TABLE df", df)

    def test_create_rejected(self, df):
        with pytest.raises(ValueError, match="CREATE"):
            run_sql("SELECT * FROM df; CREATE TABLE evil (x INT)", df)

    def test_invalid_column_raises(self, df):
        with pytest.raises(RuntimeError, match="DuckDB"):
            run_sql("SELECT nonexistent_column FROM df", df)

    def test_aggregate_query(self, df):
        result = run_sql(
            "SELECT category, COUNT(*) as cnt, AVG(value) as avg_val FROM df GROUP BY category",
            df,
        )
        assert "cnt" in result.columns
        assert len(result) == 2


class TestSchemaValidation:

    def test_identical_schema_ok(self, df):
        result = validate_schema(df, df.clone())
        assert result.ok

    def test_added_column_allowed_by_default(self, df):
        out = df.with_columns(pl.lit(1).alias("new_col"))
        result = validate_schema(df, out, allow_add_columns=True)
        assert result.ok
        assert "new_col" in result.added_columns

    def test_added_column_rejected_when_disallowed(self, df):
        out = df.with_columns(pl.lit(1).alias("new_col"))
        result = validate_schema(df, out, allow_add_columns=False)
        assert not result.ok

    def test_removed_column_rejected_by_default(self, df):
        out = df.drop("category")
        result = validate_schema(df, out, allow_remove_columns=False)
        assert not result.ok
        assert "category" in result.removed_columns

    def test_removed_column_allowed_when_configured(self, df):
        out = df.drop("category")
        result = validate_schema(df, out, allow_remove_columns=True)
        assert result.ok

    def test_dtype_change_tracked(self, df):
        out = df.with_columns(pl.col("value").cast(pl.Float32))
        result = validate_schema(df, out)
        assert any(col == "value" for col, _, _ in result.changed_dtypes)

    def test_empty_output_flagged(self, df):
        empty = df.filter(pl.lit(False))
        result = validate_schema(df, empty)
        assert not result.ok
        assert any("empty" in str(v).lower() for v in result.violations)


class TestEndToEndWorkflow:
    def test_full_happy_path(self, df):
        """Clean code → run_sample ok → apply returns transformed data."""
        code = _code("""\
            out = df.with_columns(
                (pl.col('value') * 2).alias('value_doubled')
            )
            return out
        """)
        sample_result = run_sample(code, df)
        assert sample_result.ok, sample_result.error_message
        assert sample_result.preview is not None
        assert "value_doubled" in sample_result.preview.columns
        assert "added_columns" in sample_result.schema_diff
        assert "value_doubled" in sample_result.schema_diff["added_columns"]

        full_output = apply(code, df)
        assert "value_doubled" in full_output.columns
        assert full_output["value_doubled"][0] == 0.0 * 2

    def test_step1_blocks_dangerous_code(self, df):
        code = "import os\n" + _code("return df")
        result = run_sample(code, df)
        assert not result.ok
        assert result.failed_step == 1

    def test_step2_catches_runtime_error(self, df):
        code = _code("raise ValueError('runtime boom'); return df")
        result = run_sample(code, df)
        assert not result.ok
        assert result.failed_step == 2

    def test_step3_catches_schema_violation(self, df):
        code = _code("return df.drop('category')")
        cfg = SandboxConfig(allow_remove_columns=False)
        result = run_sample(code, df, cfg)
        assert not result.ok
        assert result.failed_step == 3

    def test_schema_diff_populated(self, df):
        code = _code("return df.with_columns(pl.lit('x').alias('extra'))")
        result = run_sample(code, df)
        assert result.ok
        assert result.schema_diff["added_columns"] == ["extra"]
        assert result.schema_diff["row_count_in"] == len(df)
        assert result.schema_diff["row_count_out"] == len(df)

    def test_resource_stats_populated(self, df):
        code = _code("return df")
        result = run_sample(code, df)
        assert result.ok
        assert result.resource_stats is not None
        assert result.resource_stats.execution_time_s >= 0