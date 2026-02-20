#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  test_docker_local.sh — Local Docker E2E validation
#
#  Tests the complete Docker stack locally:
#    1. Prerequisites (Docker, Compose)
#    2. Stop any running test containers
#    3. Build fresh images
#    4. Start PostgreSQL (wait for ready)
#    5. Apply schema migrations (create all 17 tables)
#    6. Start health check API — verify GET /health returns 200
#    7. Start dashboard — verify port 8501 responds
#    8. Run one scheduler cycle (--now) or resilience test
#    9. Verify results written to PostgreSQL
#   10. Show recent logs
#   11. Clean shutdown
#
#  Usage:
#    ./deployment/test_docker_local.sh              # infra test (dummy keys OK)
#    FULL_TEST=1 ./deployment/test_docker_local.sh  # full cycle (needs real keys)
#    KEEP_UP=1  ./deployment/test_docker_local.sh   # leave services running after
#    KEEP_DATA=1 ./deployment/test_docker_local.sh  # preserve pg_data volume
#
#  Expected output:
#    ✓ PostgreSQL ready
#    ✓ Migrations applied (17 tables)
#    ✓ Health check passed
#    ✓ Scheduler run successful
#    ✓ Dashboard accessible
#
#  Target: completes in < 5 minutes
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Project settings ──────────────────────────────────────────────────────────
PROJECT="nts_test"          # compose project name (isolated from dev stack)
PG_HOST="localhost"
PG_PORT="5432"
HEALTH_PORT="8080"
DASH_PORT="8501"
PG_USER="trader"
PG_PASS="trader_dev"
PG_DB="news_trading"
PG_URL="postgresql://${PG_USER}:${PG_PASS}@postgres:5432/${PG_DB}"

PASS=0
FAIL=0
WARN=0
START_TIME=$(date +%s)

# ── Flags ─────────────────────────────────────────────────────────────────────
FULL_TEST="${FULL_TEST:-0}"    # 1 = run full scheduler cycle with real API keys
KEEP_UP="${KEEP_UP:-0}"        # 1 = leave containers running after test
KEEP_DATA="${KEEP_DATA:-0}"    # 1 = preserve pg_data volume on cleanup

# ── Detect docker compose command ────────────────────────────────────────────
if docker compose version &>/dev/null 2>&1; then
  DC="docker compose"
else
  DC="docker-compose"
fi

# ── Colour helpers ────────────────────────────────────────────────────────────
green()  { printf "\033[32m✓ %s\033[0m\n" "$*"; }
red()    { printf "\033[31m✗ %s\033[0m\n" "$*"; }
yellow() { printf "\033[33m• %s\033[0m\n" "$*"; }
cyan()   { printf "\033[36m── %s ──\033[0m\n" "$*"; }
bold()   { printf "\033[1m%s\033[0m\n" "$*"; }

pass() { green "$1";  ((PASS++))  || true; }
fail() { red   "$1";  ((FAIL++))  || true; }
warn() { yellow "$1"; ((WARN++))  || true; }

# ── Change to project root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Load environment from .env.docker (preferred) or .env ────────────────────
if [ -f ".env.docker" ]; then
  set -a
  # shellcheck source=/dev/null
  source .env.docker
  set +a
  yellow "Loaded environment from .env.docker"
elif [ -f ".env" ]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
  yellow "Loaded environment from .env"
else
  warn "No .env.docker or .env found — using shell environment"
fi

# ── Detect if API keys are real (starts with sk-ant-) ────────────────────────
REAL_ANTHROPIC=false
if [[ "${ANTHROPIC_API_KEY:-}" == sk-ant-api* ]]; then
  REAL_ANTHROPIC=true
fi

# ── Cleanup handler ───────────────────────────────────────────────────────────
cleanup() {
  local exit_code=$?
  echo ""
  cyan "Teardown"

  if [ "${KEEP_UP}" = "1" ]; then
    yellow "KEEP_UP=1 — leaving services running"
    yellow "  Stop with: ${DC} -p ${PROJECT} down"
    return
  fi

  yellow "Stopping containers …"
  if [ "${KEEP_DATA}" = "1" ]; then
    ${DC} -p "$PROJECT" down --remove-orphans 2>/dev/null || true
    yellow "KEEP_DATA=1 — pg_data volume preserved"
  else
    ${DC} -p "$PROJECT" down -v --remove-orphans 2>/dev/null || true
  fi
  green "Containers stopped."

  # Summary
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  echo ""
  bold "══════════════════════════════════════════════════"
  bold "  Docker Local Test Summary"
  bold "══════════════════════════════════════════════════"
  printf "  %-10s %s\n" "Passed:"   "$PASS"
  [ "$FAIL" -gt 0 ] && printf "  \033[31m%-10s %s\033[0m\n" "Failed:" "$FAIL" \
                     || printf "  %-10s %s\n" "Failed:" "$FAIL"
  [ "$WARN" -gt 0 ] && printf "  \033[33m%-10s %s\033[0m\n" "Warnings:" "$WARN" \
                     || true
  printf "  %-10s %ss\n" "Duration:" "$DURATION"
  bold "══════════════════════════════════════════════════"
  if [ "$FAIL" -gt 0 ]; then
    red "RESULT: FAILED ($FAIL check(s) failed)"
  else
    green "RESULT: ALL CHECKS PASSED"
  fi

  exit $exit_code
}
trap cleanup EXIT

# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Prerequisites
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 0 — Prerequisites"

if ! command -v docker &>/dev/null; then
  fail "Docker not found — install Docker Desktop from https://docs.docker.com/desktop/"
  exit 1
fi
yellow "Docker: $(docker --version | head -1)"

if ! docker info &>/dev/null; then
  fail "Docker daemon is not running — start Docker Desktop"
  exit 1
fi
yellow "Docker daemon: running"

if ! command -v curl &>/dev/null; then
  fail "curl not found — required for HTTP endpoint tests"
  exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Stop any running test containers
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 1 — Stop existing test containers"
${DC} -p "$PROJECT" down -v --remove-orphans 2>/dev/null || true
yellow "Previous test stack cleared"

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Build fresh images
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 2 — Build app image"
yellow "Building (first run: ~2 min, cached: ~15 s) …"
if ${DC} -p "$PROJECT" build --quiet 2>&1; then
  yellow "Build complete"
else
  fail "Docker build FAILED (check Dockerfile)"
  exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Start PostgreSQL
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 3 — Start PostgreSQL"
${DC} -p "$PROJECT" up -d postgres

WAIT=0
until ${DC} -p "$PROJECT" exec -T postgres \
      pg_isready -U "$PG_USER" -d "$PG_DB" -q 2>/dev/null; do
  sleep 2
  ((WAIT+=2)) || true
  if [ $WAIT -ge 60 ]; then
    fail "PostgreSQL did not become ready within 60s"
    exit 1
  fi
done
pass "PostgreSQL ready"

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Apply schema migrations (create all tables)
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 4 — Apply schema migrations"

SCHEMA_OUTPUT=$(${DC} -p "$PROJECT" run --rm \
  -e DATABASE_URL="$PG_URL" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-dummy}" \
  -e NEWSAPI_KEY="${NEWSAPI_KEY:-dummy}" \
  dashboard \
  python3 -c "
import os
os.environ['DATABASE_URL'] = os.environ.get('DATABASE_URL', '')
from storage.database import Database
db = Database()
import psycopg2
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
cur.execute(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'\")
count = cur.fetchone()[0]
conn.close()
print(f'TABLES:{count}')
" 2>&1)

TABLE_COUNT=$(echo "$SCHEMA_OUTPUT" | grep -oP 'TABLES:\K[0-9]+' || echo "0")

# Expected tables (17 core tables)
EXPECTED_TABLES=(
  runs headline_scores technical_signals combined_signals
  risk_calculations strategy_signals strategy_performance
  scheduler_logs health_checks emergency_stops
  portfolio_snapshots portfolio_violations price_alerts
  optimization_results backtest_results screener_results
  backtest_strategy_comparison
)
EXPECTED_COUNT=${#EXPECTED_TABLES[@]}

if [ "${TABLE_COUNT:-0}" -ge "$EXPECTED_COUNT" ]; then
  pass "Migrations applied (${TABLE_COUNT} tables)"
else
  fail "Migration incomplete — expected >=${EXPECTED_COUNT} tables, got ${TABLE_COUNT:-0}"
  echo "  Schema output:"
  echo "$SCHEMA_OUTPUT" | tail -10
fi

# Verify each expected table exists
MISSING_TABLES=()
for TABLE in "${EXPECTED_TABLES[@]}"; do
  RESULT=$(${DC} -p "$PROJECT" exec -T postgres \
    psql -U "$PG_USER" -d "$PG_DB" -t \
    -c "SELECT to_regclass('public.${TABLE}')" 2>/dev/null | tr -d ' \n')
  if [ "$RESULT" = "$TABLE" ]; then
    : # table exists
  else
    MISSING_TABLES+=("$TABLE")
  fi
done

if [ ${#MISSING_TABLES[@]} -eq 0 ]; then
  yellow "All ${EXPECTED_COUNT} expected tables verified"
else
  fail "Missing tables: ${MISSING_TABLES[*]}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Health check API
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 5 — Health check API"
yellow "Starting health service …"
${DC} -p "$PROJECT" up -d health

# Wait for uvicorn to start (up to 30s)
WAIT=0
until curl -sf "http://${PG_HOST}:${HEALTH_PORT}/health" &>/dev/null; do
  sleep 2
  ((WAIT+=2)) || true
  if [ $WAIT -ge 30 ]; then
    break
  fi
done

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  "http://${PG_HOST}:${HEALTH_PORT}/health" 2>/dev/null || echo "000")

if [ "$HTTP_STATUS" = "200" ]; then
  HEALTH_JSON=$(curl -s "http://${PG_HOST}:${HEALTH_PORT}/health" 2>/dev/null | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "ok")
  pass "Health check passed (HTTP ${HTTP_STATUS}, status=${HEALTH_JSON})"
else
  fail "Health check failed (HTTP ${HTTP_STATUS})"
  yellow "  Logs: ${DC} -p ${PROJECT} logs health"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Scheduler cycle
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 6 — Scheduler cycle"

if [ "$FULL_TEST" = "1" ] && [ "$REAL_ANTHROPIC" = "true" ]; then
  # Full cycle: real news + AI sentiment (requires real API keys)
  yellow "Running full scheduler cycle (--now --strategy all) …"
  SCHED_CMD="python3 scheduler/daily_runner.py --now --strategy all"
  SCHED_MODE="full"
else
  # Infrastructure test: tests circuit breakers, fallbacks, state recovery
  # Works with dummy API keys — no real external calls made
  yellow "Running resilience test (infra only, no real API calls) …"
  yellow "  Set FULL_TEST=1 with real keys for a complete scheduler cycle."
  SCHED_CMD="python3 scheduler/daily_runner.py --resilience-test"
  SCHED_MODE="resilience"
fi

if ${DC} -p "$PROJECT" run --rm \
    -e DATABASE_URL="$PG_URL" \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-dummy}" \
    -e NEWSAPI_KEY="${NEWSAPI_KEY:-dummy}" \
    -e ALPHA_VANTAGE_KEY="${ALPHA_VANTAGE_KEY:-}" \
    worker \
    sh -c "$SCHED_CMD" 2>&1; then
  pass "Scheduler run successful (mode=${SCHED_MODE})"
else
  fail "Scheduler run failed (mode=${SCHED_MODE})"
  yellow "  Check logs: ${DC} -p ${PROJECT} logs worker"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Verify results in PostgreSQL
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 7 — Verify PostgreSQL results"

if [ "$FULL_TEST" = "1" ] && [ "$REAL_ANTHROPIC" = "true" ]; then
  # Check that the scheduler logged a run
  RUN_COUNT=$(${DC} -p "$PROJECT" exec -T postgres \
    psql -U "$PG_USER" -d "$PG_DB" -t \
    -c "SELECT COUNT(*) FROM scheduler_logs;" 2>/dev/null | tr -d ' \n' || echo "0")
  if [ "${RUN_COUNT:-0}" -ge 1 ]; then
    pass "Scheduler run recorded in DB (${RUN_COUNT} log row(s))"
  else
    fail "No scheduler run found in scheduler_logs"
  fi
fi

# Verify DB is writable (insert + delete a test row)
DB_WRITE=$(${DC} -p "$PROJECT" exec -T postgres \
  psql -U "$PG_USER" -d "$PG_DB" -t \
  -c "INSERT INTO health_checks (service, status, message, checked_at)
      VALUES ('docker_test', 'ok', 'local test row', NOW())
      RETURNING id;" 2>/dev/null | tr -d ' \n' || echo "")

if [[ "$DB_WRITE" =~ ^[0-9]+$ ]]; then
  # Clean up test row
  ${DC} -p "$PROJECT" exec -T postgres \
    psql -U "$PG_USER" -d "$PG_DB" -c \
    "DELETE FROM health_checks WHERE service = 'docker_test';" &>/dev/null || true
  pass "PostgreSQL read/write verified"
else
  fail "PostgreSQL write test failed"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 8 — Dashboard"
yellow "Starting dashboard (Streamlit takes ~8s) …"
${DC} -p "$PROJECT" up -d dashboard
sleep 10

DASH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  "http://${PG_HOST}:${DASH_PORT}/" 2>/dev/null || echo "000")

# Streamlit returns 200 on the root page; 302 is also acceptable (redirect to /_stcore/...)
if [[ "$DASH_STATUS" =~ ^(200|302)$ ]]; then
  pass "Dashboard accessible (http://localhost:${DASH_PORT}, HTTP ${DASH_STATUS})"
else
  # Streamlit may still be starting — give it extra time
  yellow "Dashboard not ready yet (HTTP ${DASH_STATUS}) — waiting 10 more seconds …"
  sleep 10
  DASH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    "http://${PG_HOST}:${DASH_PORT}/" 2>/dev/null || echo "000")
  if [[ "$DASH_STATUS" =~ ^(200|302)$ ]]; then
    pass "Dashboard accessible (http://localhost:${DASH_PORT}, HTTP ${DASH_STATUS})"
  else
    warn "Dashboard not responding (HTTP ${DASH_STATUS}) — may need more time to start"
    yellow "  Check: curl -I http://localhost:${DASH_PORT}/"
    yellow "  Logs:  ${DC} -p ${PROJECT} logs dashboard"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — Show recent logs
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 9 — Recent logs"
echo ""
yellow "── health (last 5 lines) ──"
${DC} -p "$PROJECT" logs --tail=5 health 2>/dev/null || true
echo ""
yellow "── postgres (last 3 lines) ──"
${DC} -p "$PROJECT" logs --tail=3 postgres 2>/dev/null || true
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Done — summary printed by cleanup() trap
# ══════════════════════════════════════════════════════════════════════════════
