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


def generate(cfg: CodegenConfig, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_file = out_dir / "run_pipeline.py"
    pipeline_file.write_text(_render_pipeline(cfg), encoding="utf-8")
    files = [pipeline_file]

    if cfg.with_tests:
        test_file = out_dir / "test_pipeline.py"
        test_file.write_text(_render_tests(cfg), encoding="utf-8")
        files.append(test_file)

    return files


def _render_pipeline(cfg: CodegenConfig) -> str:
    ingest_src = cfg.ingest.get("source", "local_file")
    ingest_opts = json.dumps(cfg.ingest, indent=4)
    steps_repr = json.dumps(cfg.preprocess, indent=4)
    output_opts = json.dumps(cfg.output, indent=4)
    fmt = cfg.output.get("format", "parquet")
    out_path = cfg.output.get("path", "output." + fmt)

    step_blocks = "\n".join(_render_step_block(i, s) for i, s in enumerate(cfg.preprocess))
    imports = _collect_imports(cfg)

    return f"""\
\"\"\"
Auto-generated pipeline.
Source : {ingest_src}
Steps  : {len(cfg.preprocess)}
Output : {out_path} ({fmt})

Do not edit the constants section - regenerate via dp pipeline export instead.
\"\"\"

{imports}

INGEST_CONFIG: dict = {ingest_opts}
PREPROCESS_STEPS: list = {steps_repr}
OUTPUT_CONFIG: dict = {output_opts}

DRY_RUN_ROWS = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_data(dry_run: bool) -> pl.DataFrame:
    \"\"\"Load source data according to INGEST_CONFIG.\"\"\"
    source = INGEST_CONFIG.get("source", "local_file")
    if source == "local_file":
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
            raise ValueError(f"Unsupported file extension: {{ext}}")
    elif source == "postgresql":
        import asyncpg
        df = asyncio.run(_read_postgres(INGEST_CONFIG))
    elif source == "mysql":
        import asyncmy
        df = asyncio.run(_read_mysql(INGEST_CONFIG))
    elif source == "s3":
        import boto3, io
        df = _read_s3(INGEST_CONFIG)
    else:
        raise ValueError(f"Unsupported source: {{source}}")

    if dry_run:
        df = df.head(DRY_RUN_ROWS)
    return df


async def _read_postgres(cfg: dict) -> pl.DataFrame:
    conn = await asyncpg.connect(
        host=cfg["host"], port=cfg.get("port", 5432),
        database=cfg["database"], user=cfg["user"], password=cfg["password"],
    )
    try:
        sql = cfg.get("sql") or f"SELECT * FROM {{cfg['table']}}"
        rows = await conn.fetch(sql)
        return pl.DataFrame([dict(r) for r in rows])
    finally:
        await conn.close()


async def _read_mysql(cfg: dict) -> pl.DataFrame:
    conn = await asyncmy.connect(
        host=cfg["host"], port=cfg.get("port", 3306),
        db=cfg["database"], user=cfg["user"], password=cfg["password"],
    )
    try:
        async with conn.cursor() as cur:
            sql = cfg.get("sql") or f"SELECT * FROM {{cfg['table']}}"
            await cur.execute(sql)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
            return pl.DataFrame([dict(zip(cols, r)) for r in rows])
    finally:
        conn.close()


def _read_s3(cfg: dict) -> pl.DataFrame:
    import boto3, io
    s3 = boto3.client(
        "s3",
        aws_access_key_id=cfg.get("aws_access_key_id"),
        aws_secret_access_key=cfg.get("aws_secret_access_key"),
        region_name=cfg.get("region", "us-east-1"),
        endpoint_url=cfg.get("endpoint_url"),
    )
    buf = io.BytesIO()
    s3.download_fileobj(cfg["bucket"], cfg["key"], buf)
    buf.seek(0)
    ext = Path(cfg["key"]).suffix.lower()
    if ext == ".parquet":
        return pl.read_parquet(buf)
    return pl.read_csv(buf)


{step_blocks}

def _write_output(df: pl.DataFrame, dry_run: bool) -> None:
    \"\"\"Write output according to OUTPUT_CONFIG. Skipped in dry-run mode.\"\"\"
    if dry_run:
        log.info("[dry-run] output skipped. Preview (10 rows):")
        print(df.head(10))
        return
    fmt = OUTPUT_CONFIG.get("format", "parquet")
    path = Path(OUTPUT_CONFIG.get("path", f"output.{{fmt}}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.write_csv(path, separator=OUTPUT_CONFIG.get("delimiter", ","))
    elif fmt == "parquet":
        df.write_parquet(path, compression=OUTPUT_CONFIG.get("compression", "snappy"))
    elif fmt in ("excel", "xlsx"):
        df.write_excel(path, worksheet=OUTPUT_CONFIG.get("sheet_name", "Sheet1"))
    elif fmt == "json":
        df.write_json(path)
    elif fmt == "jsonl":
        df.write_ndjson(path)
    elif fmt in ("arrow", "feather"):
        import pyarrow.feather as feather
        feather.write_feather(df.to_arrow(), str(path))
    else:
        raise ValueError(f"Unsupported output format: {{fmt}}")
    log.info(f"Wrote {{len(df)}} rows to {{path}}")


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
    base = [
        "import asyncio",
        "import logging",
        "from pathlib import Path",
        "",
        "import polars as pl",
    ]
    src = cfg.ingest.get("source", "")
    if src == "postgresql":
        base.append("import asyncpg")
    if src == "mysql":
        base.append("import asyncmy")
    if src == "s3":
        base.extend(["import boto3", "import io"])
    return "\n".join(base)


def _render_step_block(idx: int, step: dict[str, Any]) -> str:
    step_type = step.get("step", step.get("type", "unknown"))
    columns = step.get("columns", [])
    params = {k: v for k, v in step.get("params", {}).items()}
    func_name = f"_step_{idx:03d}_{step_type}"

    col_repr = repr(columns)
    params_repr = repr(params)

    body = _build_step_body(step_type, columns, params)

    return f'''\
def {func_name}(df: pl.DataFrame) -> pl.DataFrame:
    """Step {idx}: {step_type} on columns {columns or "all"}."""
    cols: list = {col_repr}
    params: dict = {params_repr}
    active_cols = cols or [c for c in df.columns if df[c].dtype.is_numeric()] if {_is_numeric_step(step_type)} else cols or df.columns
{textwrap.indent(body, "    ")}
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
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    df = df.with_columns(pl.col(col).clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr))"""

    if step_type == "clip_zscore":
        return """\
threshold = params.get("threshold", 3.0)
for col in active_cols:
    mean, std = df[col].mean(), df[col].std()
    if std:
        df = df.with_columns(pl.col(col).clip(mean - threshold * std, mean + threshold * std))"""

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
        return """\
order_map = params.get("order", {})
for col in active_cols:
    order = order_map.get(col) or df[col].drop_nulls().unique().sort().to_list()
    mapping = {v: i for i, v in enumerate(order)}
    df = df.with_columns(pl.col(col).replace(mapping, default=None).cast(pl.Int32))"""

    if step_type == "select_columns":
        return "df = df.select(active_cols)"

    if step_type == "drop_columns":
        return "df = df.drop(active_cols)"

    if step_type == "rename_columns":
        return "df = df.rename(params.get('mapping', {}))"

    if step_type == "cast_dtype":
        return """\
dtype_map = {
    "int8": pl.Int8, "int16": pl.Int16, "int32": pl.Int32, "int64": pl.Int64,
    "float32": pl.Float32, "float64": pl.Float64,
    "str": pl.String, "string": pl.String, "bool": pl.Boolean,
    "date": pl.Date, "datetime": pl.Datetime,
}
for col, dtype_str in params.get("casts", {}).items():
    target = dtype_map[dtype_str.lower()]
    df = df.with_columns(pl.col(col).cast(target))"""

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


def _render_tests(cfg: CodegenConfig) -> str:
    fmt = cfg.output.get("format", "parquet")
    out_path = cfg.output.get("path", "output." + fmt)

    return f"""\
\"\"\"
Auto-generated test scaffold for run_pipeline.
Run with: python -m pytest test_pipeline.py -v
\"\"\"

import importlib
import unittest
from pathlib import Path

import polars as pl

import run_pipeline as _mod


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