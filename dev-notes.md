# datapill - Local Development & Testing

This guide walks you through spinning up the full test environment, seeding all data sources, and verifying each connector end-to-end. Follow the steps in order.

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.11+ |
| Docker + Docker Compose | v2+ |
| uv | latest |

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 1. Clone & install

```bash
git clone <repo-url> datapill
cd datapill

# create virtualenv and install all dependencies (including dev extras)
uv sync --all-extras
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# verify the CLI is available
dp --version
```

---

## 2. Start test services

All external services (PostgreSQL, MySQL, MinIO, Kafka) run via Docker Compose.

```bash
docker compose -f docker-compose.test.yml up -d --wait
```

The `--wait` flag blocks until every service's healthcheck passes. Expected output:

```
✔ Container datapill-postgres-1   Healthy
✔ Container datapill-mysql-1      Healthy
✔ Container datapill-minio-1      Healthy
✔ Container datapill-minio-init-1 Exited (0)
✔ Container datapill-kafka-1      Healthy
```

> **Port map** - Postgres `5433`, MySQL `3307`, MinIO API `9000` / Console `9001`, Kafka `9092`. If any port conflicts with a local service, update `docker-compose.test.yml` and the corresponding config in `tests/fixtures/configs/`.

---

## 3. Seed all data sources

The seed script populates every connector with 100-row `employees` and 5-row `departments` tables (or equivalent objects for S3 / Kafka / local files).

```bash
# seed everything at once
python scripts/seed.py

# or seed individual sources only
python scripts/seed.py --only postgres mysql
python scripts/seed.py --only sqlite local
python scripts/seed.py --only s3 kafka
```

Expected output:

```
============================================================
datapill  -  seeding test data
============================================================
→ Configs  ✓ tests/fixtures/configs/  (8 config files)
           ✓ tests/fixtures/ops/       (4 ops files)

Generating DataFrames…
  generating employees (150k + ~750 dupes)…
  generating products (5k)…
  generating customers (50k)…
  generating orders (200k)…
  generating transactions (300k)…
  generating events (500k)…
  generating logs (100k)…
  generating reviews (80k)…
  generating inventory…
  generating departments…
  ✓ total rows across all tables: 1,385,750+

→ Local files  employees, products, customers, orders, transactions,
               events, logs, reviews, inventory, departments
               → csv + parquet + ndjson per table
               → employees_by_dept/  (8 parquet partition files)
               → orders_by_month/    (~60 parquet partition files)
               → employees_dirty.csv (injected nulls/bad values)
→ SQLite    ✓ tests/fixtures/test.db  (9 tables)
→ PostgreSQL ✓ (9 tables)
→ MySQL     ✓ (9 tables)
→ MinIO/S3  ✓ data/{table}.{csv,parquet,ndjson} per table
→ Kafka     ✓ multiple topics seeded
============================================================
done - 1,385,750+ total rows seeded across 10 tables
```

Fixture configs land in `tests/fixtures/configs/`. Preprocess ops examples land in `tests/fixtures/ops/`. All subsequent `dp` commands reference these files.

> **Tables seeded:** `employees` (150k+dupes), `products` (5k), `customers` (50k), `orders` (200k), `transactions` (300k), `events` (500k), `logs` (100k), `reviews` (80k), `inventory` (~12k), `departments` (8).

> **New configs:** `local_dirty.json` (writable base path for testing clean ops), `rest_api.json` (JSONPlaceholder base URL).

---

## 4. Verify connections

Run `ingest check` against each source before doing a full ingest. All should print `✔ connected`.

```bash
dp ingest check local \
    --config tests/fixtures/configs/local.json \
    --path   data.parquet

dp ingest check sqlite \
    --config tests/fixtures/configs/sqlite.json

dp ingest check postgres \
    --config tests/fixtures/configs/postgres.json

dp ingest check mysql \
    --config tests/fixtures/configs/mysql.json

dp ingest check s3 \
    --config tests/fixtures/configs/s3.json

dp ingest check kafka \
    --config tests/fixtures/configs/kafka.json
```

---

## 5. Run ingestions

### Local file

```bash
# full ingest - parquet
dp ingest run local \
    --config      tests/fixtures/configs/local.json \
    --path        data.parquet \
    --materialize \
    --schema

# ingest from any seeded table file
dp ingest run local \
    --config      tests/fixtures/configs/local.json \
    --path        orders.parquet \
    --materialize \
    --schema

# sample a large table
dp ingest run local \
    --config      tests/fixtures/configs/local.json \
    --path        events.csv \
    --sample \
    --sample-size 5000 \
    --schema

# ingest dirty CSV for testing clean ops downstream
dp ingest run local \
    --config      tests/fixtures/configs/local_dirty.json \
    --path        employees_dirty.csv \
    --schema
```

### SQLite

```bash
dp ingest run sqlite \
    --config      tests/fixtures/configs/sqlite.json \
    --table       employees \
    --materialize \
    --schema

# custom query
dp ingest run sqlite \
    --config tests/fixtures/configs/sqlite.json \
    --query  "SELECT dept, COUNT(*) AS n FROM employees GROUP BY dept"
```

### PostgreSQL

```bash
dp ingest run postgres \
    --config      tests/fixtures/configs/postgres.json \
    --table       employees \
    --materialize \
    --schema

# streaming with batch-size
dp ingest run postgres \
    --config      tests/fixtures/configs/postgres.json \
    --table       employees \
    --batch-size  50 \
    --materialize
```

### MySQL

```bash
dp ingest run mysql \
    --config      tests/fixtures/configs/mysql.json \
    --table       employees \
    --materialize \
    --schema
```

### Amazon S3 (MinIO)

```bash
# ingest a parquet file from the test bucket
dp ingest run s3 \
    --config tests/fixtures/configs/s3.json \
    --path   data/employees.parquet \
    --materialize

# ingest CSV
dp ingest run s3 \
    --config tests/fixtures/configs/s3.json \
    --path   data/employees.csv \
    --sample \
    --sample-size 30
```

### Kafka

```bash
# consume all 100 seeded messages
dp ingest run kafka \
    --config tests/fixtures/configs/kafka.json \
    --topic  employees \
    --materialize

# sample - stops after 20 messages
dp ingest run kafka \
    --config      tests/fixtures/configs/kafka.json \
    --topic       employees \
    --sample \
    --sample-size 20
```

Each successful ingest prints a `run_id` (e.g. `a1b2c3d4`). Save the IDs you want to profile in the next section.

---

## 6. Preprocess an artifact

Use a `run_id` from an ingest step (or a previous preprocess step) as the parent. Ops files live in `tests/fixtures/ops/`.

### Run with inline ops

```bash
# trim + lowercase email, drop nulls, log-transform salary
dp preprocess run <INGEST_RUN_ID> \
    --op clean.drop_null:cols=email,salary \
    --op parse.trim:cols=full_name,email \
    --op parse.lower:cols=email \
    --op transform.log_transform:cols=salary:base=log10 \
    --materialize \
    --schema
```

### Run with an ops file

```bash
# clean employees - drops nulls, clips age/salary, deduplicates on email
dp preprocess run <INGEST_RUN_ID> \
    --ops tests/fixtures/ops/clean_employees.json \
    --materialize \
    --schema

# feature engineering - normalize, bin age, one-hot encode dept
dp preprocess run <INGEST_RUN_ID> \
    --ops tests/fixtures/ops/feature_engineering.json \
    --materialize

# clean orders - parse datetime, extract parts, deduplicate
dp preprocess run <ORDERS_INGEST_RUN_ID> \
    --ops tests/fixtures/ops/clean_orders.json \
    --materialize

# aggregate orders by country × status
dp preprocess run <ORDERS_INGEST_RUN_ID> \
    --ops tests/fixtures/ops/aggregate_orders.json \
    --materialize
```

### Preview without materializing

```bash
# inspect the first 20 rows after ops are applied (join ops are skipped)
dp preprocess preview <PREPROCESS_RUN_ID> --rows 20
```

### Materialize a previously dry-run artifact

```bash
dp preprocess materialize <PREPROCESS_RUN_ID> --schema
```

### List available ops

```bash
# all groups
dp preprocess ops

# filter to a single group
dp preprocess ops --group clean
dp preprocess ops --group transform
dp preprocess ops --group compose
```

Each successful preprocess run prints a `run_id`. You can chain preprocess steps (pass a preprocess `run_id` as the parent of another preprocess run), then profile the final artifact.

## 7. Profile an artifact

Use the `run_id` from an ingest step above. Substitute `<RUN_ID>` accordingly.

```bash
# full profile - histograms, correlations, pattern detection, materialized JSON
dp profile run <RUN_ID> \
    --mode full \
    --correlation pearson \
    --correlation-threshold 0.2 \
    --schema

# faster summary-only profile (no histogram / correlation output)
dp profile run <RUN_ID> --mode summary

# profile with random sampling (useful for the 1k-row local fixture)
dp profile run <RUN_ID> \
    --mode            full \
    --sample-strategy random \
    --sample-size     500
```

### Inspect the profile

```bash
# full breakdown: dataset stats, column table, correlations, warnings
dp profile show <PROFILE_RUN_ID>

# warnings only
dp profile warnings <PROFILE_RUN_ID>

# filter to a specific severity
dp profile warnings <PROFILE_RUN_ID> --severity error
dp profile warnings <PROFILE_RUN_ID> --severity warn
```

---

## 8. Manage artifacts

```bash
# list all artifacts (most recent first)
dp artifact list

# filter by pipeline
dp artifact list --pipeline ingest
dp artifact list --pipeline profile --limit 5

# inspect a single artifact
dp artifact show <RUN_ID>

# trace lineage from a profile artifact back to raw ingest
dp artifact lineage <PROFILE_RUN_ID>

# disk usage summary
dp artifact usage

# delete a single run (skips confirmation prompt)
dp artifact delete <RUN_ID> --yes

# keep only the 3 most recent ingest artifacts, purge the rest
dp artifact purge --pipeline ingest --keep 3 --yes

# purge sample-only artifacts across all pipelines
dp artifact purge --samples-only --yes
```

---

## 9. Tear down

```bash
# stop containers and remove volumes
docker compose -f docker-compose.test.yml down -v

# remove generated fixtures and artifact store
rm -rf tests/fixtures/test.db tests/fixtures/data*.csv tests/fixtures/data*.parquet tests/fixtures/data*.ndjson
rm -rf .datapill
```

---

## Troubleshooting

**`dp: command not found`** - make sure the virtualenv is activated (`source .venv/bin/activate`) and the package is installed (`uv sync`).

**Service healthcheck timeout** - MySQL can take ~30 s on first boot. Re-run `docker compose -f docker-compose.test.yml up -d --wait` if any service shows `Waiting`.

**Kafka seed fails with `NoBrokersAvailable`** - Kafka's advertised listener is `localhost:9092`. The seed script must run on the host, not inside a container. Wait a few seconds after `--wait` returns and retry.

**`artifact not found` when profiling** - the `run_id` is case-sensitive and must belong to a pipeline that `profile` accepts (`ingest` or `preprocess`). Run `dp artifact list` to confirm the ID and pipeline name.

**`cannot delete <id>: has N child artifact(s)`** - the artifact you are trying to delete has downstream artifacts (e.g. a profile run). Either delete children first (leaf to root), or re-run with `--cascade` to remove the entire subtree in one go.

**Port conflict** - edit the `ports` mapping in `docker-compose.test.yml` and update the matching `host`/`port` fields in the corresponding `tests/fixtures/configs/*.json` file, then re-seed.