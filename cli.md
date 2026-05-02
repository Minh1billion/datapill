# datapill - CLI Reference & Test Guide

> For agents: read this top-to-bottom before running any command.

---

## Overview

`datapill` is a data ingestion CLI. Entry point is `dp` (or `uv run datapill` in dev).  
Two top-level command groups: **`ingest`** and **`artifact`**.

```
dp ingest   - pull data from a source
dp artifact - inspect and manage stored run artifacts
```

All state is stored under `.datapill/` in the working directory.  
All config files are JSON. Fixtures live in `tests/fixtures/configs/`.

---

## Prerequisites

```bash
# 1. Start test services
docker compose -f docker-compose.test.yml up -d --wait

# 2. Seed test data
uv run python scripts/seed.py

# 3. Confirm CLI works
uv run datapill --help
```

Supported sources after seed: `postgres`, `mysql`, `sqlite`, `s3`, `local`, `kafka`, `rest`

---

## `dp ingest` - Ingest Commands

### `ingest run` - Pull data from a source

```
dp ingest run <source> --config <path.json> [options]
```

| Flag | Description |
|---|---|
| `--config / -c` | **(required)** Path to connector config JSON |
| `--table / -t` | Table name - postgres, mysql, sqlite |
| `--query / -q` | Raw SQL - postgres, mysql, sqlite |
| `--topic` | Kafka topic |
| `--path / -p` | File path - s3, local |
| `--endpoint / -e` | REST endpoint path |
| `--params` | Path to params JSON - REST only |
| `--sample` | Read a sample instead of full data |
| `--sample-size` | Rows to sample (default: 10 000) |
| `--batch-size` | Rows per streaming batch |
| `--materialize / -m` | Write output to `.parquet` artifact on disk |
| `--schema` | Print column schema after ingest |
| `--store` | Artifact store directory (default: `.datapill`) |

**Rules by source:**

| Source | Required option |
|---|---|
| `postgres` `mysql` `sqlite` | `--table` or `--query` |
| `kafka` | `--topic` - always add `--sample` to avoid blocking |
| `s3` `local` | `--path` |
| `rest` | `--endpoint` |

### `ingest check` - Test connection only

```
dp ingest check <source> --config <path.json>
```

### `ingest sources` - List all supported sources

```
dp ingest sources
```

---

## `dp artifact` - Artifact Commands

Each `ingest run` produces an artifact with a unique `run_id`.  
Use these commands to inspect, trace, and clean them up.

### `artifact list`

```
dp artifact list [--pipeline ingest] [--limit 20] [--store .datapill]
```

Shows: run_id, pipeline, timestamp, row count, sample flag, materialized flag.

### `artifact show <run_id>`

```
dp artifact show <run_id>
```

Shows full metadata: schema, options, path (if materialized), parent lineage.

### `artifact lineage <run_id>`

```
dp artifact lineage <run_id>
```

Traces the parent chain of a run as an indented tree.

### `artifact delete <run_id>`

```
dp artifact delete <run_id> [--yes]
```

Deletes one artifact. Prompts for confirmation unless `--yes`.

### `artifact purge`

```
dp artifact purge [--pipeline ingest] [--keep 3] [--samples-only] [--yes]
```

Bulk delete. `--keep N` preserves the N most recent. `--samples-only` targets only sample runs.

### `artifact usage`

```
dp artifact usage
```

Prints total artifact count and disk usage.

---

## Full Test Flow

Run in order from top to bottom. Each step depends on the previous one succeeding.

---

### Step 0 - Start services & seed

```bash
docker compose -f docker-compose.test.yml up -d --wait
uv run python scripts/seed.py
uv run datapill --help
dp ingest sources
```

**Expected:** sources prints 7 lines: `postgres mysql sqlite s3 local kafka rest`

---

### Step 1 - Check connections

```bash
dp ingest check postgres --config tests/fixtures/configs/postgres.json
dp ingest check mysql    --config tests/fixtures/configs/mysql.json
dp ingest check sqlite   --config tests/fixtures/configs/sqlite.json
dp ingest check s3       --config tests/fixtures/configs/s3.json
dp ingest check kafka    --config tests/fixtures/configs/kafka.json
dp ingest check rest     --config tests/fixtures/configs/rest_api.json
```

**Expected per command:** `+ connected  X.Xms` then `ok`

> **REST:** uses `jsonplaceholder.typicode.com` - public, no auth required.  
> Update `tests/fixtures/configs/rest_api.json` before running this step:
> ```json
> {
>   "base_url": "https://jsonplaceholder.typicode.com",
>   "pagination_type": "page",
>   "page_param": "_page",
>   "page_size_param": "_limit",
>   "page_size": 20,
>   "results_key": null
> }
> ```

---

### Step 2 - Ingest sample (no disk write)

```bash
# postgres - sample 20 rows, print schema
dp ingest run postgres \
  --config tests/fixtures/configs/postgres.json \
  --table employees --sample --sample-size 20 --schema

# postgres - raw query
dp ingest run postgres \
  --config tests/fixtures/configs/postgres.json \
  --query "SELECT * FROM departments" --schema

# mysql - sample
dp ingest run mysql \
  --config tests/fixtures/configs/mysql.json \
  --table employees --sample --sample-size 20 --schema

# sqlite - full table (100 rows)
dp ingest run sqlite \
  --config tests/fixtures/configs/sqlite.json \
  --table employees --schema

# sqlite - raw query
dp ingest run sqlite \
  --config tests/fixtures/configs/sqlite.json \
  --query "SELECT dept, COUNT(*) as n FROM employees GROUP BY dept" --schema

# s3 - csv
dp ingest run s3 \
  --config tests/fixtures/configs/s3.json \
  --path data/employees.csv --schema

# s3 - parquet
dp ingest run s3 \
  --config tests/fixtures/configs/s3.json \
  --path data/employees.parquet --schema

# s3 - ndjson
dp ingest run s3 \
  --config tests/fixtures/configs/s3.json \
  --path data/employees.ndjson --schema

# local - csv
dp ingest run local \
  --config tests/fixtures/configs/local.json \
  --path data.csv --schema

# local - parquet
dp ingest run local \
  --config tests/fixtures/configs/local.json \
  --path data.parquet --schema

# local - large file 1k rows, streaming batch 100
dp ingest run local \
  --config tests/fixtures/configs/local.json \
  --path data_1k.csv --batch-size 100 --schema

# kafka - ALWAYS use --sample to avoid blocking
dp ingest run kafka \
  --config tests/fixtures/configs/kafka.json \
  --topic employees --sample --sample-size 30 --schema

# rest - /posts (200 records, paginated)
dp ingest run rest \
  --config tests/fixtures/configs/rest_api.json \
  --endpoint posts --schema

# rest - sample 20 rows
dp ingest run rest \
  --config tests/fixtures/configs/rest_api.json \
  --endpoint posts --sample --sample-size 20 --schema

# rest - different endpoint
dp ingest run rest \
  --config tests/fixtures/configs/rest_api.json \
  --endpoint comments --sample --sample-size 30 --schema
```

**Expected per command:** `✓ N rows  M cols` + schema if `--schema` is set

---

### Step 3 - Full ingest + materialize (write parquet to disk)

```bash
# postgres - full 100 rows, write parquet
dp ingest run postgres \
  --config tests/fixtures/configs/postgres.json \
  --table employees --materialize --schema

# mysql - full, write parquet
dp ingest run mysql \
  --config tests/fixtures/configs/mysql.json \
  --table employees --materialize

# sqlite - full, write parquet
dp ingest run sqlite \
  --config tests/fixtures/configs/sqlite.json \
  --table employees --materialize

# s3 - csv, write parquet
dp ingest run s3 \
  --config tests/fixtures/configs/s3.json \
  --path data/employees.csv --materialize

# local - 1k rows, batch + materialize
dp ingest run local \
  --config tests/fixtures/configs/local.json \
  --path data_1k.csv --batch-size 200 --materialize --schema

# rest - full /posts, write parquet
dp ingest run rest \
  --config tests/fixtures/configs/rest_api.json \
  --endpoint /posts --materialize --schema
```

**Expected:** output includes `path  artifacts/<run_id>/data.parquet`

---

### Step 4 - Inspect artifact store

```bash
# List all artifacts
dp artifact list

# Show full detail of a materialized artifact (get run_id from list above)
dp artifact show <run_id>

# Trace lineage
dp artifact lineage <run_id>

# Show disk usage
dp artifact usage
```

**Expected `artifact list`:** table with more than 10 rows, `materialized` column shows `y` for step 3 runs

---

### Step 5 - Manage artifacts

```bash
# Delete one specific sample artifact (get run_id from list)
dp artifact delete <run_id> --yes

# Purge all sample artifacts
dp artifact purge --samples-only --yes

# Verify after purge
dp artifact list
dp artifact usage

# Purge all, keep 3 most recent
dp artifact purge --keep 3 --yes

# Confirm only 3 remain
dp artifact list
```

**Expected at end:** `artifact list` shows exactly 3 rows, `artifact usage` reflects correct count

---

## Output Reference

A successful `ingest run` prints:

```
* connecting to postgres
  + connected  4.2ms
  ✓ 100 rows  8 cols

  run   abc123...
  ref   ingest/abc123
```

A successful `artifact list` prints a table with columns:  
`run id | pipeline | when | rows | sample | materialized`

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Kafka hangs | Always use `--sample` with kafka |
| REST `results_key` error | Set `"results_key": null` in rest_api.json when using jsonplaceholder |
| S3 `NoSuchBucket` | Wait for `minio-init` container to finish |
| `artifact not found` | Run `artifact list` first to get a valid `run_id` |
| Import errors | Run `uv sync` to install all dependencies |
| Stale data | Delete `.datapill/` and re-seed |