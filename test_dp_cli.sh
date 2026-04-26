#!/usr/bin/env bash
set -euo pipefail

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0

log()  { echo -e "${BOLD}[$(date +%T)]${NC} $*"; }
ok()   { echo -e "  ${GREEN}[OK]${NC} $*"; ((PASS++)) || true; }
fail() { echo -e "  ${RED}[FAIL]${NC} $*"; ((FAIL++)) || true; }
warn() { echo -e "  ${YELLOW}⚠${NC} $*"; ((SKIP++)) || true; }
sep()  { echo -e "\n${CYAN}══════════════════════════════════════════${NC}"; }

WORK_DIR="${TMPDIR:-/tmp}/dp_test_$$"
if command -v cygpath &>/dev/null; then
  WORK_DIR=$(cygpath -m "$WORK_DIR")
fi
mkdir -p "$WORK_DIR"

OUT_DIR="$WORK_DIR/artifacts"
mkdir -p "$OUT_DIR"

SAMPLE_CSV="$WORK_DIR/sample.csv"
SAMPLE_PARQUET="$WORK_DIR/sample.parquet"
SEED_PY="$WORK_DIR/seed.py"
PG_CONFIG="$WORK_DIR/pg.json"
MY_CONFIG="$WORK_DIR/my.json"
S3_CONFIG="$WORK_DIR/s3.json"
KAFKA_CONFIG="$WORK_DIR/kafka.json"
REST_CONFIG="$WORK_DIR/rest.json"
RUN_CONFIG="$WORK_DIR/run.json"
PG_QUERY_CONFIG="$WORK_DIR/pg_query.json"
LAST_OUT="$WORK_DIR/last_out.txt"

sep
log "Checking environment"

if ! command -v dp &>/dev/null; then
  echo -e "${RED}Error: 'dp' not found. Please activate virtualenv first.${NC}"
  exit 1
fi
ok "dp CLI: $(command -v dp)"
ok "Python: $(python --version 2>&1)"
log "Temp directory: $WORK_DIR"

sep
log "Generating sample data"

export DP_WORK_DIR="$WORK_DIR"
cat > "$SEED_PY" << 'PYEOF'
import polars as pl
import os

w = os.environ["DP_WORK_DIR"]
df = pl.DataFrame({
    "id": list(range(1, 101)),
    "name": ["user_" + str(i) for i in range(1, 101)],
    "score": [round(i * 1.5 + 0.1, 2) for i in range(1, 101)],
    "active": [i % 2 == 0 for i in range(1, 101)],
})
df.write_csv(w + "/sample.csv")
df.write_parquet(w + "/sample.parquet")
print("OK:" + str(len(df)))
PYEOF

python "$SEED_PY" > "$WORK_DIR/seed_out.txt" 2>&1
if grep -q "OK:" "$WORK_DIR/seed_out.txt"; then
  ok "Created sample.csv & sample.parquet (100 rows)"
else
  fail "Failed to create sample data"
  cat "$WORK_DIR/seed_out.txt"
  exit 1
fi

sep
log "Creating config files"

cat > "$PG_CONFIG" << 'EOF'
{
  "host": "localhost",
  "port": 5433,
  "database": "testdb",
  "user": "testuser",
  "password": "testpass"
}
EOF

cat > "$PG_QUERY_CONFIG" << 'EOF'
{
  "host": "localhost",
  "port": 5433,
  "database": "testdb",
  "user": "testuser",
  "password": "testpass",
  "sql": "SELECT * FROM test_table LIMIT 20"
}
EOF

cat > "$MY_CONFIG" << 'EOF'
{
  "host": "localhost",
  "port": 3307,
  "database": "testdb",
  "user": "testuser",
  "password": "testpass"
}
EOF

cat > "$S3_CONFIG" << 'EOF'
{
  "aws_access_key_id": "minioadmin",
  "aws_secret_access_key": "minioadmin",
  "region": "us-east-1",
  "endpoint_url": "http://localhost:9000",
  "bucket": "testbucket"
}
EOF

cat > "$KAFKA_CONFIG" << 'EOF'
{
  "bootstrap_servers": "localhost:9092",
  "group_id": "dp-test",
  "auto_offset_reset": "earliest",
  "default_topic": "dp-test-topic"
}
EOF

python -c "
import json
with open('$REST_CONFIG', 'w') as f:
    json.dump({
        'base_url': 'https://jsonplaceholder.typicode.com',
        'response_path': '',
        'default_endpoint': '/todos'
    }, f)
"

cat > "$RUN_CONFIG" << EOF
{
  "source": "local_file",
  "connector": {},
  "query": {"path": "$SAMPLE_PARQUET"},
  "ingest": {"batch_size": 50}
}
EOF

ok "All config files created"

check_service() {
  local host=$1 port=$2
  timeout 2 bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null && return 0 || return 1
}

if check_service localhost 5433; then
  log "Preparing PostgreSQL test table"
  docker exec -i $(docker ps -qf "name=postgres") psql -U testuser -d testdb <<-'EOSQL' 2>/dev/null
    CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, name TEXT);
    INSERT INTO test_table (name) VALUES ('test1'), ('test2'), ('test3') ON CONFLICT DO NOTHING;
EOSQL
  if [ $? -eq 0 ]; then
    ok "PostgreSQL test table created"
  else
    warn "Could not create test table in PostgreSQL"
  fi
fi

run_test() {
  local label="$1"; shift
  dp "$@" --out "$OUT_DIR" > "$LAST_OUT" 2>&1 || true
  
  if grep -qiE "rows.*[1-9][0-9]*|Ingest complete" "$LAST_OUT"; then
    ok "$label"
  elif grep -qiE "error|failed|exception|cannot connect|does not exist" "$LAST_OUT"; then
    if grep -qiE "jsonplaceholder.typicode.comc" "$LAST_OUT"; then
      fail "$label (URL corrupted - script bug)"
    else
      fail "$label"
    fi
    sed 's/^/    /' "$LAST_OUT" | head -10
  else
    fail "$label"
    sed 's/^/    /' "$LAST_OUT" | head -10
  fi
}

run_connector() {
  local label="$1"; shift
  dp "$@" > "$LAST_OUT" 2>&1 || true
  
  if grep -qiE "Connection OK|Column|Schema:" "$LAST_OUT"; then
    ok "$label"
  else
    fail "$label"
    sed 's/^/    /' "$LAST_OUT" | head -10
  fi
}

sep
log "GROUP 1: local_file"

log "1.1  ingest CSV"
run_test "ingest local_file CSV" ingest --source local_file --path "$SAMPLE_CSV" --batch-size 30

log "1.2  ingest Parquet"
run_test "ingest local_file Parquet" ingest --source local_file --path "$SAMPLE_PARQUET" --batch-size 50

log "1.3  connector test"
run_connector "connector test local_file" connector test --source local_file --path "$SAMPLE_CSV"

log "1.4  connector schema"
run_connector "connector schema local_file" connector schema --source local_file --path "$SAMPLE_CSV"

sep
log "GROUP 2: profile"

log "2.1  profile full"
dp profile --input "$SAMPLE_PARQUET" --mode full --out "$OUT_DIR" > "$LAST_OUT" 2>&1 || true
if grep -qiE "Profile ID" "$LAST_OUT"; then
  ok "profile full"
else
  fail "profile full"
  sed 's/^/    /' "$LAST_OUT" | head -10
fi

log "2.2  profile summary + random sampling"
dp profile --input "$SAMPLE_PARQUET" --mode summary \
  --sample-strategy random --sample-size 50 --out "$OUT_DIR" > "$LAST_OUT" 2>&1 || true
if grep -qiE "Profile ID" "$LAST_OUT"; then
  ok "profile summary+sampling"
else
  fail "profile summary+sampling"
  sed 's/^/    /' "$LAST_OUT" | head -10
fi

sep
log "GROUP 3: dp run (pipeline JSON)"

log "3.1  dp run local_file"
dp run "$RUN_CONFIG" --out "$OUT_DIR" > "$LAST_OUT" 2>&1 || true
if grep -qiE "Ingest complete.*[1-9][0-9]* rows" "$LAST_OUT"; then
  ok "dp run pipeline"
else
  fail "dp run pipeline"
  sed 's/^/    /' "$LAST_OUT" | head -10
fi

sep
log "GROUP 4: REST (jsonplaceholder.typicode.com)"

log "4.1  connector test"
run_connector "connector test rest" connector test --source rest --config "$REST_CONFIG"

log "4.2  connector schema"
run_connector "connector schema rest" connector schema --source rest --config "$REST_CONFIG" --endpoint todos

log "4.3  ingest /todos"
run_test "ingest rest /todos" ingest --source rest --config "$REST_CONFIG" --endpoint todos --batch-size 10

sep
log "GROUP 5: PostgreSQL (requires Docker)"

log "5.1  connector test"
if check_service localhost 5433; then
  run_connector "connector test postgresql" connector test --source postgresql --config "$PG_CONFIG"
else
  warn "PostgreSQL not running, skipping test"
fi

log "5.2  connector schema test_table"
if check_service localhost 5433; then
  run_connector "connector schema postgresql" connector schema --source postgresql --config "$PG_CONFIG" --table test_table
else
  warn "PostgreSQL not running, skipping schema test"
fi

log "5.3  ingest test_table using config with sql"
if check_service localhost 5433; then
  run_test "ingest postgresql test_table" ingest --source postgresql --config "$PG_QUERY_CONFIG"
else
  warn "PostgreSQL not running, skipping ingest test"
fi

sep
log "GROUP 6: MySQL (requires Docker)"

log "6.1  connector test"
if check_service localhost 3307; then
  run_connector "connector test mysql" connector test --source mysql --config "$MY_CONFIG"
else
  warn "MySQL not running, skipping test"
fi

sep
log "GROUP 7: S3 / MinIO (requires Docker)"

log "7.1  connector test"
if check_service localhost 9000; then
  run_connector "connector test s3" connector test --source s3 --config "$S3_CONFIG"
else
  warn "MinIO not running, skipping test"
fi

sep
log "GROUP 8: Kafka (requires Docker)"

log "8.1  connector test"
if check_service localhost 9092; then
  run_connector "connector test kafka" connector test --source kafka --config "$KAFKA_CONFIG"
else
  warn "Kafka not running, skipping test"
fi

sep
log "Cleaning up $WORK_DIR"
rm -rf "$WORK_DIR"
ok "Removed temp directory"

sep
echo ""
echo -e "${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         dp CLI SMOKE TEST RESULTS    ║${NC}"
echo -e "${BOLD}╠══════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC}  ${GREEN}PASS${NC} : ${GREEN}${PASS}${NC}"
echo -e "${BOLD}║${NC}  ${RED}FAIL${NC} : ${RED}${FAIL}${NC}"
echo -e "${BOLD}║${NC}  ${YELLOW}SKIP${NC} : ${YELLOW}${SKIP}${NC}"
echo -e "${BOLD}╚══════════════════════════════════════╝${NC}"
echo ""

if [[ $FAIL -gt 0 ]]; then
  echo -e "${RED}$FAIL test(s) failed — see logs above.${NC}"
  exit 1
else
  echo -e "${GREEN}All tests PASSED [OK]${NC}"
  exit 0
fi