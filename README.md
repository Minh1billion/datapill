# 🗂️ Data Pill

**A modular, open-source data preparation CLI for developers, data analysts, data scientists, and data engineers.**

[![PyPI version](https://img.shields.io/pypi/v/datapill?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/datapill/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-pytest-blue?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![SaaS](https://img.shields.io/badge/SaaS-coming%20soon-orange)](https://minh1billion.github.io/datapill/)

datapill gives you a single command-line tool to ingest, profile, classify, preprocess, and export data - from any source, to any destination - with a clean pipeline model and full artifact tracking. A hosted SaaS is coming soon.

---

## 💡 Why Data Pill?

Most data prep work happens in ad-hoc scripts, notebooks, or half-finished internal tools. datapill replaces that chaos with a reproducible, composable pipeline that works the same way whether you're exploring a CSV on your laptop or processing millions of rows from Kafka in production.

| | If you are… | datapill helps you… |
|---|---|---|
| 🧑‍💻 | **Developer** | Build data pipelines from config files, run them in CI, generate standalone Python scripts |
| 📊 | **Data Analyst** | Profile any dataset in seconds, detect nulls, distributions, and anomalies without writing code |
| 🤖 | **Data Scientist** | Classify columns by semantic type, preprocess features (scaling, imputation, encoding) with one command |
| ⚙️ | **Data Engineer** | Stream-ingest from Postgres, MySQL, S3, Kafka, REST APIs; write back with upsert support |

---

## 📦 Installation

```bash
pip install datapill
```

Requires Python 3.11+.

This installs the core CLI (~200 MB) with rule-based classification, profiling, preprocessing, and export. No ML model is required.

### With embedding support

To use `--mode embedding` or `--mode hybrid` in `dp classify`, install the ML extras (~3.5 GB, includes PyTorch and sentence-transformers):

```bash
pip install "datapill[ml]"
```

> **Note:** Without `[ml]`, `hybrid` mode still works — it runs rule-based classification for all columns and skips the embedding fallback. Only columns that would otherwise be sent to the embedding model are affected.

Verify the install:

```bash
dp --help
```

---

## 🚀 Quick Start

### 1. Ingest a local CSV

```bash
dp ingest --source local_file --path data/sales.csv
```

### 2. Profile the dataset

```bash
dp profile --input <run_id>
```

### 3. Classify columns by semantic type

```bash
dp classify --input <run_id> --mode hybrid
```

### 4. Preprocess with a pipeline config

```bash
dp preprocess --input <run_id> --pipeline pipeline.json
```

### 5. Export to Parquet

```bash
dp export --input <run_id> --format parquet --out-path output/result.parquet
```

---

## 🛠️ Commands

### `dp ingest`

Stream data from any supported source into the artifact store.

```bash
dp ingest --source postgresql --config pg.json --table orders
dp ingest --source s3 --config s3.json --url s3://my-bucket/data.parquet
dp ingest --source kafka --config kafka.json --topic events --max-records 10000
dp ingest --source rest --config api.json --endpoint /users
dp ingest --source local_file --path data/sales.csv --limit 50000
```

**Supported sources:** `local_file` · `postgresql` · `mysql` · `s3` · `rest` · `kafka`

**Supported formats (local / S3):** `csv` · `parquet` · `json` · `jsonl` · `excel`

---

### `dp profile`

Compute a full statistical profile of any ingested dataset.

```bash
dp profile --input <run_id>
dp profile --input <run_id> --mode summary
dp profile --input <run_id> --sample-strategy random --sample-size 100000
dp profile --input <run_id> --correlation spearman
```

**What you get:**
- Per-column: null rate, distinct count, min/max/mean/median, std, skewness, kurtosis, percentiles, histogram
- Top value frequencies with percentages
- Pattern detection: email, URL, phone, UUID, ISO date
- Correlation matrix (Pearson or Spearman) for all numeric columns
- Warnings: `HIGH_NULL_RATE`, `CONSTANT_COLUMN`, `SKEWED_DISTRIBUTION`, `HIGH_CARDINALITY`, `POTENTIAL_IDENTIFIER`

---

### `dp classify`

Classify every column in a dataset by its semantic type - automatically.

```bash
dp classify --input <run_id> --mode hybrid
dp classify --input <run_id> --mode rule_based --threshold 0.65
dp classify --input <run_id> --overrides '{"age": "numerical_continuous", "y": "target_label"}'
```

**Modes:**

| Mode | How it works | Requires |
|---|---|---|
| `rule_based` | Regex patterns on column names + dtype heuristics. Fast, zero ML dependencies. | core |
| `embedding` | Semantic similarity via `sentence-transformers` (`all-MiniLM-L6-v2`) against anchor texts per type. | `datapill[ml]` |
| `hybrid` | Rule-based first; embedding kicks in only for ambiguous or unknown columns. | `datapill[ml]` for full accuracy |

> **Without `[ml]`:** `hybrid` mode runs entirely on rule-based logic. Columns that cannot be resolved by rules are returned as `unknown` instead of being sent to the embedding model.

**Semantic types detected:** `identifier` · `numerical_continuous` · `numerical_discrete` · `categorical_nominal` · `categorical_ordinal` · `text_freeform` · `text_structured` · `datetime` · `boolean` · `geospatial` · `embedding` · `target_label`

---

### `dp preprocess`

Apply a preprocessing pipeline defined in a JSON config file.

```bash
dp preprocess --input <run_id> --pipeline pipeline.json
dp preprocess --input <run_id> --pipeline pipeline.json --dry-run
dp preprocess --input <run_id> --pipeline pipeline.json --checkpoint
```

**Pipeline config format:**

```json
{
  "steps": [
    { "type": "impute_mean",      "scope": { "columns": ["age", "income"] } },
    { "type": "clip_iqr",         "scope": { "columns": ["income"] } },
    { "type": "standard_scaler",  "scope": { "columns": ["age", "income"] } },
    { "type": "onehot",           "scope": { "columns": ["category"] } },
    { "type": "drop_missing",     "scope": { "columns": [] } }
  ]
}
```

**Available steps:**

| Category | Steps |
|---|---|
| Missing values | `impute_mean` · `impute_median` · `impute_mode` · `drop_missing` |
| Outliers | `clip_iqr` · `clip_zscore` |
| Scaling | `standard_scaler` · `minmax_scaler` · `robust_scaler` |
| Encoding | `onehot` · `ordinal` |
| Structure | `select_columns` · `drop_columns` · `rename_columns` · `cast_dtype` · `deduplicate` |
| Custom | `custom_python` (sandboxed via RestrictedPython) |

`--dry-run` runs on the first 1,000 rows and prints a preview without saving any artifact.  
`--checkpoint` saves the DataFrame after each step so you can resume from any point.

---

### `dp export`

Export a processed dataset to a file or write it back to a database or S3.

```bash
# Export to file
dp export --input <run_id> --format parquet --out-path output/result.parquet

# Write back to PostgreSQL (upsert)
dp export --input <run_id> --format parquet \
  --connector pg.json --write-mode upsert --primary-keys id

# Write to S3
dp export --input <run_id> --format csv --connector s3.json
```

**Write modes:** `replace` · `append` · `upsert`  
**Output formats:** `csv` · `parquet` · `json` · `jsonl` · `excel` · `arrow`

---

### `dp pipeline export`

Generate a standalone Python script from a preprocess pipeline artifact — no datapill dependency required at runtime.

```bash
dp pipeline export -i <run_id> -s local_file --path data.csv
dp pipeline export -i <run_id> -s postgresql -c pg.json
dp pipeline export -i <run_id> -s local_file --path data.csv --with-tests
dp pipeline export -i <run_id> --out-dir ./generated
```

**What you get:**
- Preprocessing steps reconstructed from the saved config artifact
- Ingest configuration merged into a single self-contained script
- A `run_<name>.py` entry point with a `--dry-run` flag
- An optional `test_<name>.py` scaffold (with `--with-tests`)

**Generated files:**

| File | Description |
|---|---|
| `run_<name>.py` | Main pipeline script — runs without datapill |
| `test_<name>.py` | pytest scaffold (only with `--with-tests`) |

**Options:**

| Option | Description |
|---|---|
| `--input`, `-i` | `run_id` or preprocess artifact ID |
| `--source`, `-s` | Connector type: `local_file` · `postgresql` · `mysql` · `s3` |
| `--ingest-config`, `-c` | Connector JSON config (same as `dp ingest --config`) |
| `--path` | File path (`local_file`) |
| `--table` | Table name (`postgresql` \| `mysql`) |
| `--url` | S3 URL |
| `--format`, `-f` | Output format: `csv` · `parquet` · `json` · `jsonl` · `excel` |
| `--out-path` | Output path hard-coded into the generated script |
| `--name`, `-n` | Base name for generated files, e.g. `orders` → `run_orders.py` |
| `--compression` | `snappy` · `zstd` · `gzip` (parquet only) |
| `--with-tests` | Also generate `test_<name>.py` |
| `--out-dir`, `-o` | Directory to write generated files (default: `generated/`) |

**Run the generated pipeline:**

```bash
python generated/run_<name>.py --dry-run
python generated/run_<name>.py
```

**Run the generated tests:**

```bash
python -m pytest generated/test_<name>.py -v
```

> **Note:** `dp pipeline export` requires a preprocess artifact saved without `--dry-run`. If only a dry-run artifact exists, re-run `dp preprocess` without that flag first.

---

### `dp connector`

Inspect and interact with any connector directly - without running a full pipeline.

```bash
dp connector test     --source postgresql --config pg.json
dp connector schema   --source postgresql --config pg.json --table orders
dp connector upload   --source s3 --config s3.json --src-path data.csv --dest-url s3://bucket/data.csv
dp connector download --source s3 --config s3.json --url s3://bucket/data.csv --out-path ./data.csv
dp connector list     --source s3 --config s3.json --prefix input/
dp connector exec     --source postgresql --config pg.json --sql "DELETE FROM orders WHERE status='cancelled'"
dp connector truncate --source postgresql --config pg.json --table orders
dp connector produce  --source kafka --config kafka.json --topic events --file records.json
```

---

### `dp list`

List all artifacts in the store.

```bash
dp list
dp list --feature ingest
dp list --limit 50
```

---

### `dp run`

Run a full ingest + profile pipeline from a single config file.

```bash
dp run pipeline.json
```

**Config format:**

```json
{
  "source": "postgresql",
  "connector": { "host": "localhost", "database": "mydb", "user": "u", "password": "p" },
  "query": { "table": "orders" },
  "ingest": { "batch_size": 10000 },
  "profile": { "mode": "full", "correlation": "pearson" }
}
```

---

## 🔌 Connector Configuration

All connectors are configured via JSON files passed with `--config`.

### PostgreSQL / MySQL

```json
{
  "host": "localhost",
  "port": 5432,
  "database": "mydb",
  "user": "myuser",
  "password": "mypassword"
}
```

### S3

```json
{
  "aws_access_key_id": "AKIA...",
  "aws_secret_access_key": "...",
  "region": "us-east-1",
  "bucket": "my-bucket"
}
```

### Kafka

```json
{
  "bootstrap_servers": ["localhost:9092"],
  "group_id": "datapill",
  "value_format": "json",
  "security_protocol": "PLAINTEXT"
}
```

SASL/SSL is supported - add `sasl_mechanism`, `sasl_username`, `sasl_password`, and `ssl_cafile` as needed.

### REST API

```json
{
  "base_url": "https://api.example.com",
  "headers": { "Authorization": "Bearer <token>" },
  "response_path": "data",
  "pagination": {
    "type": "offset",
    "limit": 100,
    "limit_param": "limit",
    "offset_param": "offset"
  }
}
```

Pagination modes: `offset` · `cursor` · `link_header`

---

## 🗃️ Artifact Store

Every pipeline run produces **artifacts** - Parquet files and JSON metadata - stored locally and tracked in a registry.

```
src/datapill/artifacts/
├── registry.json
├── a1b2c3d4_ingest_output.parquet
├── a1b2c3d4_ingest_schema.json
├── e5f6g7h8_profile_detail.json
└── e5f6g7h8_profile_summary.json
```

You can reference any artifact by its `run_id` (short 8-char hex) or full `artifact_id`. datapill resolves ambiguity automatically using feature-aware priority rules.

Change the artifact directory with `--out`:

```bash
dp ingest --source local_file --path data.csv --out /my/artifacts
```

---

## 🏗️ Architecture

```
datapill/
├── cli/            # Typer CLI - entry point for all commands
├── connectors/     # Source adapters (local, PG, MySQL, S3, REST, Kafka)
├── core/           # PipelineContext, ProgressEvent, FeaturePipeline interface
├── executor/       # Sandboxed code execution (RestrictedPython + Docker)
├── features/
│   ├── ingest/     # Stream ingestion → Parquet artifacts
│   ├── profile/    # Statistical profiling + correlation
│   ├── classify/   # Semantic type classification (rule-based + embedding)
│   ├── preprocess/ # Step-based transformation pipeline
│   └── export/     # File export + DB/S3 write-back + code generation
└── storage/        # ArtifactStore - registry, save/load, resolve
```

Every feature implements the same `FeaturePipeline` interface: `validate → plan → execute`. Pipelines emit async `ProgressEvent` streams so the CLI can render live progress bars.

---

## 🐍 Custom Python Steps (Sandboxed)

You can write arbitrary Python transformation logic and run it safely inside the preprocess pipeline.

```python
# my_transform.py
def transform(df):
    return df.with_columns(
        (pl.col("revenue") / pl.col("units")).alias("avg_price")
    )
```

```json
{
  "steps": [
    {
      "type": "custom_python",
      "scope": { "columns": [] },
      "params": { "code": "<contents of my_transform.py>", "func": "transform" }
    }
  ]
}
```

Custom code is validated by an AST analyzer (banned imports, banned builtins, dunder access) before execution. Two sandbox backends are available:

- **RestrictedPython** - in-process, low overhead, suitable for most use cases
- **Docker** - full container isolation (`--network none`, read-only FS, memory + CPU limits), for untrusted code

---

## 🧑‍💻 Development Setup

```bash
git clone https://github.com/your-org/datapill.git
cd datapill
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

For full embedding support in development:

```bash
pip install -e ".[ml,dev]"
```

Run the tests:

```bash
pytest
pytest -m "not integration"     # skip tests that require Docker services
pytest --cov=datapill
```

Lint:

```bash
ruff check src/
ruff format src/
```

---

## 🗺️ Roadmap

- [ ] Web UI / dashboard for artifact browsing and profile visualization
- [ ] Custom step registry - register and share reusable step plugins
- [ ] datapill SaaS - hosted pipelines, scheduling, collaboration, and monitoring
- [ ] dbt integration - use datapill as a pre-processing layer before dbt models
- [ ] Great Expectations integration - attach data quality assertions to any pipeline step

---

## 🤝 Contributing

Contributions are welcome. Please open an issue before submitting a large pull request so we can discuss the approach.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes with tests
4. Run `ruff check` and `pytest` before pushing
5. Open a pull request against `main`

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

*datapill SaaS - hosted pipelines, scheduling, and collaboration - coming soon.*