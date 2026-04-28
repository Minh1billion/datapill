import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CodegenConfig:
    ingest: dict[str, Any]
    preprocess: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    with_tests: bool = False


def generate(cfg: CodegenConfig, out_dir: Path, name: str | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"_{name}" if name else ""
    pipeline_file = out_dir / f"run{stem}.py"
    pipeline_file.write_text(_render_pipeline(cfg), encoding="utf-8")
    files = [pipeline_file]

    if cfg.with_tests:
        test_file = out_dir / f"test{stem}.py"
        test_file.write_text(_render_tests(cfg, stem), encoding="utf-8")
        files.append(test_file)

    return files


def _render_pipeline(cfg: CodegenConfig) -> str:
    src = cfg.ingest.get("source", "local_file")
    fmt = cfg.output.get("format", "parquet")
    out_path = cfg.output.get("path", "output." + fmt)

    return f"""\
\"\"\"
Auto-generated pipeline.
Source : {src}
Steps  : {len(cfg.preprocess)}
Output : {out_path} ({fmt})

Do not edit the constants section - regenerate via dp pipeline export instead.
\"\"\"

{_collect_imports(cfg)}

INGEST_CONFIG: dict = {json.dumps(cfg.ingest, indent=4)}
OUTPUT_CONFIG: dict = {json.dumps(cfg.output, indent=4)}

DRY_RUN_ROWS = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


{_render_load_data(src)}

{"".join(_render_step_block(i, s) for i, s in enumerate(cfg.preprocess))}
{_render_write_output(fmt)}

def run_pipeline(dry_run: bool = False) -> pl.DataFrame:
    \"\"\"Entry point. Set dry_run=True to preview without writing output.\"\"\"
    log.info(f"Starting pipeline (dry_run={{dry_run}})")

    try:
        df = _load_data(dry_run)
        log.info(f"Loaded {{len(df)}} rows, {{len(df.columns)}} columns")
    except Exception as exc:
        log.error(f"Load failed: {{exc}}")
        raise

{_render_step_calls(cfg.preprocess)}
    try:
        _write_output(df, dry_run)
    except Exception as exc:
        log.error(f"Write failed: {{exc}}")
        raise

    log.info("Pipeline complete")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_pipeline(dry_run=args.dry_run)
"""


def _collect_imports(cfg: CodegenConfig) -> str:
    src = cfg.ingest.get("source", "local_file")
    lines = [
        "import logging",
        "from pathlib import Path",
        "",
        "import polars as pl",
    ]
    if src == "postgresql":
        lines = ["import asyncio"] + lines + ["import asyncpg"]
    elif src == "mysql":
        lines = ["import asyncio"] + lines + ["import asyncmy"]
    elif src == "s3":
        lines += ["import boto3", "import io"]
    return "\n".join(lines)


def _render_load_data(src: str) -> str:
    if src == "local_file":
        return """\
def _load_data(dry_run: bool) -> pl.DataFrame:
    path = INGEST_CONFIG["path"]
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pl.read_csv(path)
    elif ext == ".parquet":
        df = pl.read_parquet(path)
    elif ext == ".json":
        df = pl.read_json(path)
    elif ext in (".jsonl", ".ndjson"):
        df = pl.read_ndjson(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df.head(DRY_RUN_ROWS) if dry_run else df
"""

    if src == "postgresql":
        return """\
async def _load_data_async() -> pl.DataFrame:
    conn = await asyncpg.connect(
        host=INGEST_CONFIG["host"], port=INGEST_CONFIG.get("port", 5432),
        database=INGEST_CONFIG["database"], user=INGEST_CONFIG["user"],
        password=INGEST_CONFIG["password"],
    )
    try:
        sql = INGEST_CONFIG.get("sql") or f"SELECT * FROM {INGEST_CONFIG['table']}"
        rows = await conn.fetch(sql)
        return pl.DataFrame([dict(r) for r in rows])
    finally:
        await conn.close()


def _load_data(dry_run: bool) -> pl.DataFrame:
    df = asyncio.run(_load_data_async())
    return df.head(DRY_RUN_ROWS) if dry_run else df
"""

    if src == "mysql":
        return """\
async def _load_data_async() -> pl.DataFrame:
    conn = await asyncmy.connect(
        host=INGEST_CONFIG["host"], port=INGEST_CONFIG.get("port", 3306),
        db=INGEST_CONFIG["database"], user=INGEST_CONFIG["user"],
        password=INGEST_CONFIG["password"],
    )
    try:
        async with conn.cursor() as cur:
            sql = INGEST_CONFIG.get("sql") or f"SELECT * FROM {INGEST_CONFIG['table']}"
            await cur.execute(sql)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
            return pl.DataFrame([dict(zip(cols, r)) for r in rows])
    finally:
        conn.close()


def _load_data(dry_run: bool) -> pl.DataFrame:
    df = asyncio.run(_load_data_async())
    return df.head(DRY_RUN_ROWS) if dry_run else df
"""

    if src == "s3":
        return """\
def _load_data(dry_run: bool) -> pl.DataFrame:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=INGEST_CONFIG.get("aws_access_key_id"),
        aws_secret_access_key=INGEST_CONFIG.get("aws_secret_access_key"),
        region_name=INGEST_CONFIG.get("region", "us-east-1"),
        endpoint_url=INGEST_CONFIG.get("endpoint_url"),
    )
    buf = io.BytesIO()
    s3.download_fileobj(INGEST_CONFIG["bucket"], INGEST_CONFIG["key"], buf)
    buf.seek(0)
    ext = Path(INGEST_CONFIG["key"]).suffix.lower()
    df = pl.read_parquet(buf) if ext == ".parquet" else pl.read_csv(buf)
    return df.head(DRY_RUN_ROWS) if dry_run else df
"""

    return """\
def _load_data(dry_run: bool) -> pl.DataFrame:
    raise NotImplementedError(f"Source '{INGEST_CONFIG.get('source')}' not supported in generated code")
"""


def _render_write_output(fmt: str) -> str:
    writers = {
        "csv": 'df.write_csv(path, separator=OUTPUT_CONFIG.get("delimiter", ","))',
        "parquet": 'df.write_parquet(path, compression=OUTPUT_CONFIG.get("compression", "snappy"))',
        "excel": 'df.write_excel(path, worksheet=OUTPUT_CONFIG.get("sheet_name", "Sheet1"))',
        "xlsx": 'df.write_excel(path, worksheet=OUTPUT_CONFIG.get("sheet_name", "Sheet1"))',
        "json": "df.write_json(path)",
        "jsonl": "df.write_ndjson(path)",
        "arrow": "import pyarrow.feather as feather; feather.write_feather(df.to_arrow(), str(path))",
        "feather": "import pyarrow.feather as feather; feather.write_feather(df.to_arrow(), str(path))",
    }
    write_stmt = writers.get(fmt, f'raise ValueError("Unsupported output format: {fmt}")')

    return f"""\
def _write_output(df: pl.DataFrame, dry_run: bool) -> None:
    if dry_run:
        log.info("[dry-run] output skipped. Preview (10 rows):")
        print(df.head(10))
        return
    path = Path(OUTPUT_CONFIG.get("path", "output.{fmt}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    {write_stmt}
    log.info(f"Wrote {{len(df)}} rows to {{path}}")
"""


def _render_step_block(idx: int, step: dict[str, Any]) -> str:
    step_type = step.get("step", step.get("type", "unknown"))
    columns = step.get("columns", [])
    params = {k: v for k, v in step.get("params", {}).items()}
    is_numeric = _is_numeric_step(step_type)

    if columns:
        active_cols_line = f"active_cols = {repr(columns)}"
    elif is_numeric:
        active_cols_line = "active_cols = [c for c in df.columns if df[c].dtype.is_numeric()]"
    else:
        active_cols_line = "active_cols = df.columns"

    body = _build_step_body(step_type, columns, params)
    params_line = f"    params: dict = {repr(params)}\n" if params else ""

    return f'''\
def _step_{idx:03d}_{step_type}(df: pl.DataFrame) -> pl.DataFrame:
    {active_cols_line}
{params_line}{textwrap.indent(body, "    ")}
    return df


'''


def _build_step_body(step_type: str, columns: list, params: dict) -> str:
    if step_type == "impute_mean":
        return """\
for col in active_cols:
    df = df.with_columns(pl.col(col).fill_null(df[col].mean()))"""

    if step_type == "impute_median":
        return """\
for col in active_cols:
    df = df.with_columns(pl.col(col).fill_null(df[col].median()))"""

    if step_type == "impute_mode":
        return """\
for col in active_cols:
    mode = df[col].drop_nulls().mode()
    if len(mode):
        df = df.with_columns(pl.col(col).fill_null(mode[0]))"""

    if step_type == "drop_missing":
        return "df = df.drop_nulls(subset=active_cols or None)"

    if step_type == "clip_iqr":
        return """\
for col in active_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    df = df.with_columns(pl.col(col).clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr))"""

    if step_type == "clip_zscore":
        threshold = params.get("threshold", 3.0)
        return f"""\
for col in active_cols:
    mean, std = df[col].mean(), df[col].std()
    if std:
        df = df.with_columns(pl.col(col).clip(mean - {threshold} * std, mean + {threshold} * std))"""

    if step_type == "standard_scaler":
        return """\
for col in active_cols:
    mean, std = df[col].mean(), df[col].std()
    if std:
        df = df.with_columns(((pl.col(col) - mean) / std).alias(col))"""

    if step_type == "minmax_scaler":
        return """\
for col in active_cols:
    lo, hi = df[col].min(), df[col].max()
    if hi != lo:
        df = df.with_columns(((pl.col(col) - lo) / (hi - lo)).alias(col))"""

    if step_type == "robust_scaler":
        return """\
for col in active_cols:
    median = df[col].median()
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    if iqr:
        df = df.with_columns(((pl.col(col) - median) / iqr).alias(col))"""

    if step_type == "onehot":
        return """\
for col in active_cols:
    for cat in df[col].drop_nulls().unique().sort().to_list():
        df = df.with_columns((pl.col(col) == cat).cast(pl.Int8).alias(f"{col}__{cat}"))
    df = df.drop(col)"""

    if step_type == "ordinal":
        order_map = params.get("order", {})
        return f"""\
order_map = {repr(order_map)}
for col in active_cols:
    order = order_map.get(col) or df[col].drop_nulls().unique().sort().to_list()
    mapping = {{v: i for i, v in enumerate(order)}}
    df = df.with_columns(pl.col(col).replace(mapping, default=None).cast(pl.Int32))"""

    if step_type == "select_columns":
        return "df = df.select(active_cols)"

    if step_type == "drop_columns":
        return "df = df.drop(active_cols)"

    if step_type == "rename_columns":
        mapping = params.get("mapping", {})
        return f"df = df.rename({repr(mapping)})"

    if step_type == "cast_dtype":
        casts = params.get("casts", {})
        return f"""\
dtype_map = {{
    "int8": pl.Int8, "int16": pl.Int16, "int32": pl.Int32, "int64": pl.Int64,
    "float32": pl.Float32, "float64": pl.Float64,
    "str": pl.String, "string": pl.String, "bool": pl.Boolean,
    "date": pl.Date, "datetime": pl.Datetime,
}}
for col, dtype_str in {repr(casts)}.items():
    df = df.with_columns(pl.col(col).cast(dtype_map[dtype_str.lower()]))"""

    if step_type == "deduplicate":
        return "df = df.unique(subset=active_cols or None, keep='first')"

    return f'raise NotImplementedError("step {step_type} not implemented in generated code")'


def _is_numeric_step(step_type: str) -> bool:
    return step_type in {
        "impute_mean", "impute_median", "clip_iqr", "clip_zscore",
        "standard_scaler", "minmax_scaler", "robust_scaler",
    }


def _render_step_calls(steps: list[dict]) -> str:
    if not steps:
        return ""
    lines = []
    for i, step in enumerate(steps):
        step_type = step.get("step", step.get("type", "unknown"))
        func = f"_step_{i:03d}_{step_type}"
        lines.append(f"""\
    try:
        df = {func}(df)
        log.info(f"Step {i} '{step_type}' done - {{len(df)}} rows")
    except Exception as exc:
        log.error(f"Step {i} '{step_type}' failed: {{exc}}")
        raise
""")
    return "\n".join(lines)


def _render_tests(cfg: CodegenConfig, stem: str = "") -> str:
    fmt = cfg.output.get("format", "parquet")
    out_path = cfg.output.get("path", "output." + fmt)
    module = f"run{stem}"

    return f"""\
\"\"\"
Auto-generated test scaffold for {module}.
Run with: python -m pytest {module.replace('run', 'test')}.py -v
\"\"\"

import unittest
from pathlib import Path

import polars as pl

import {module} as _mod


class TestPipelineSmoke(unittest.TestCase):
    def test_dry_run_returns_dataframe(self):
        df = _mod.run_pipeline(dry_run=True)
        self.assertIsInstance(df, pl.DataFrame)
        self.assertGreater(len(df), 0)

    def test_dry_run_row_limit(self):
        df = _mod.run_pipeline(dry_run=True)
        self.assertLessEqual(len(df), _mod.DRY_RUN_ROWS)

    def test_output_schema_not_empty(self):
        df = _mod.run_pipeline(dry_run=True)
        self.assertGreater(len(df.columns), 0)


class TestPipelineOutput(unittest.TestCase):
    def setUp(self):
        self.out_path = Path("{out_path}")

    def test_output_file_exists_after_run(self):
        _mod.run_pipeline(dry_run=False)
        self.assertTrue(self.out_path.exists(), f"Output file not found: {{self.out_path}}")

    def test_output_file_not_empty(self):
        _mod.run_pipeline(dry_run=False)
        self.assertGreater(self.out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
"""