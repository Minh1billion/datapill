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

## Minimal Test Sequence

Run these in order to verify the full stack works.

```bash
# --- connections ---
dp ingest check postgres --config tests/fixtures/configs/postgres.json
dp ingest check mysql    --config tests/fixtures/configs/mysql.json
dp ingest check sqlite   --config tests/fixtures/configs/sqlite.json
dp ingest check s3       --config tests/fixtures/configs/s3.json
dp ingest check kafka    --config tests/fixtures/configs/kafka.json

# --- ingest samples ---
dp ingest run postgres --config tests/fixtures/configs/postgres.json --table employees --sample --sample-size 20 --schema
dp ingest run mysql    --config tests/fixtures/configs/mysql.json    --table employees --sample --sample-size 20
dp ingest run sqlite   --config tests/fixtures/configs/sqlite.json   --table employees --schema
dp ingest run s3       --config tests/fixtures/configs/s3.json       --path data/employees.csv --schema
dp ingest run local    --config tests/fixtures/configs/local.json    --path data.csv --schema
dp ingest run kafka    --config tests/fixtures/configs/kafka.json    --topic employees --sample --sample-size 30 --schema

# --- materialize one ---
dp ingest run postgres --config tests/fixtures/configs/postgres.json --table employees --materialize --schema

# --- inspect artifacts ---
dp artifact list
dp artifact show <run_id>        # use run_id from output above
dp artifact usage

# --- cleanup ---
dp artifact purge --keep 3 --yes
```

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
| S3 `NoSuchBucket` | Wait for `minio-init` container to finish |
| `artifact not found` | Run `artifact list` first to get a valid `run_id` |
| Import errors | Run `uv sync` to install all dependencies |
| Stale data | Delete `.datapill/` and re-seed |