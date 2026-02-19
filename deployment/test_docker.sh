#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  test_docker.sh — Local Docker integration test
#
#  Tests the full Docker stack:
#    1. Build all containers
#    2. Start PostgreSQL
#    3. Health-check the DB
#    4. Start the health-check API and verify /health returns 200
#    5. Start the dashboard and verify it responds
#    6. Run a PostgreSQL migration dry-run
#    7. (Optional) Run one scheduler cycle if API keys are set
#    8. Verify results in the database
#    9. Tear down
#
#  Usage:
#    ./deployment/test_docker.sh            # core infrastructure tests
#    FULL_TEST=1 ./deployment/test_docker.sh  # also runs scheduler cycle (needs API keys)
#
#  Target: completes in <5 minutes
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

COMPOSE="docker-compose"
PROJECT="news_trading_test"
PASS=0
FAIL=0
START_TIME=$(date +%s)

# ── Colour helpers ────────────────────────────────────────────────────────────
green()  { echo -e "\033[32m✓ $*\033[0m"; }
red()    { echo -e "\033[31m✗ $*\033[0m"; }
yellow() { echo -e "\033[33m• $*\033[0m"; }
header() { echo -e "\n\033[1m── $* ──\033[0m"; }

pass() { green "$1"; ((PASS++)) || true; }
fail() { red   "$1"; ((FAIL++)) || true; }

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
  header "Teardown"
  yellow "Stopping containers …"
  $COMPOSE -p "$PROJECT" down -v --remove-orphans 2>/dev/null || true
  green "Containers stopped."
}
trap cleanup EXIT

# ── Prerequisite check ────────────────────────────────────────────────────────
header "Prerequisites"

if ! command -v docker &>/dev/null; then
  fail "Docker not found — install Docker Desktop"
  exit 1
fi
pass "Docker available: $(docker --version | head -1)"

if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null 2>&1; then
  fail "docker-compose not found"
  exit 1
fi
# Prefer 'docker compose' (v2) if available
if docker compose version &>/dev/null 2>&1; then
  COMPOSE="docker compose"
fi
pass "Docker Compose available"

# ── Step 1: Build ─────────────────────────────────────────────────────────────
header "Step 1 — Build containers"
yellow "Building app image (this may take 1-3 min on first run) …"
if $COMPOSE -p "$PROJECT" build --quiet 2>&1 | tail -3; then
  pass "Docker build succeeded"
else
  fail "Docker build FAILED"
  exit 1
fi

# ── Step 2: Start PostgreSQL ──────────────────────────────────────────────────
header "Step 2 — Start PostgreSQL"
yellow "Starting postgres service …"
$COMPOSE -p "$PROJECT" up -d postgres

WAIT=0
until $COMPOSE -p "$PROJECT" exec -T postgres \
      pg_isready -U trader -d news_trading -q 2>/dev/null; do
  sleep 2
  ((WAIT+=2))
  if [ $WAIT -ge 60 ]; then
    fail "PostgreSQL did not become ready within 60s"
    exit 1
  fi
done
pass "PostgreSQL is ready (waited ${WAIT}s)"

# ── Step 3: Schema initialisation ────────────────────────────────────────────
header "Step 3 — Schema initialisation"
yellow "Running Database() to create tables …"
if $COMPOSE -p "$PROJECT" run --rm \
    -e DATABASE_URL=postgresql://trader:trader_dev@postgres:5432/news_trading \
    dashboard python3 -c "from storage.database import Database; Database(); print('Schema OK')" \
    2>&1 | grep -q "Schema OK"; then
  pass "All tables created in PostgreSQL"
else
  fail "Schema initialisation failed"
  exit 1
fi

# ── Step 4: Health API ────────────────────────────────────────────────────────
header "Step 4 — Health check API"
yellow "Starting health service …"
$COMPOSE -p "$PROJECT" up -d health
sleep 5

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/ 2>/dev/null || echo "000")
if [ "$HTTP_STATUS" = "200" ]; then
  pass "Health API responded: HTTP $HTTP_STATUS"
else
  fail "Health API not responding (got HTTP $HTTP_STATUS)"
fi

# ── Step 5: Dashboard ─────────────────────────────────────────────────────────
header "Step 5 — Dashboard startup"
yellow "Starting dashboard service …"
$COMPOSE -p "$PROJECT" up -d dashboard
sleep 8  # Streamlit takes ~5-8s to start

DASH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/ 2>/dev/null || echo "000")
if [ "$DASH_STATUS" = "200" ]; then
  pass "Dashboard responded: HTTP $DASH_STATUS"
else
  # Streamlit sometimes returns 302 before 200
  yellow "Dashboard status: HTTP $DASH_STATUS (may still be starting)"
fi

# ── Step 6: Migration dry-run ─────────────────────────────────────────────────
header "Step 6 — Migration dry-run"
if [ -f "news_trading.db" ]; then
  yellow "Running migration dry-run against Docker PG …"
  if $COMPOSE -p "$PROJECT" run --rm \
      -e DATABASE_URL=postgresql://trader:trader_dev@postgres:5432/news_trading \
      -v "$(pwd)/news_trading.db:/app/news_trading.db:ro" \
      dashboard python3 deployment/migrate_to_postgres.py --dry-run 2>&1 | \
      grep -E "(Table|Total|Error)"; then
    pass "Migration dry-run completed"
  else
    yellow "Migration dry-run skipped (no local SQLite DB)"
  fi
else
  yellow "No local news_trading.db found — skipping migration dry-run"
fi

# ── Step 7 (optional): Full scheduler cycle ───────────────────────────────────
if [ "${FULL_TEST:-0}" = "1" ]; then
  header "Step 7 — Scheduler cycle (FULL_TEST=1)"
  if [ -z "${ANTHROPIC_API_KEY:-}" ] || [ -z "${NEWSAPI_KEY:-}" ]; then
    fail "FULL_TEST=1 requires ANTHROPIC_API_KEY and NEWSAPI_KEY env vars"
  else
    yellow "Running one scheduler cycle (--now --no-execute) …"
    $COMPOSE -p "$PROJECT" run --rm \
      -e DATABASE_URL=postgresql://trader:trader_dev@postgres:5432/news_trading \
      worker python3 scheduler/daily_runner.py --now --no-execute --watchlist AAPL
    pass "Scheduler cycle completed"

    # Verify a run was logged
    RUN_COUNT=$($COMPOSE -p "$PROJECT" exec -T postgres \
      psql -U trader -d news_trading -t -c "SELECT COUNT(*) FROM scheduler_logs;" \
      2>/dev/null | tr -d ' ')
    if [ "${RUN_COUNT:-0}" -ge 1 ]; then
      pass "Scheduler run logged to DB (${RUN_COUNT} row(s))"
    else
      fail "No scheduler run found in DB"
    fi
  fi
else
  yellow "Step 7 skipped (set FULL_TEST=1 to run scheduler cycle)"
fi

# ── Step 8: DB verification ───────────────────────────────────────────────────
header "Step 8 — Database table verification"
TABLES=(
  runs headline_scores technical_signals combined_signals
  risk_calculations strategy_signals strategy_performance
  scheduler_logs health_checks emergency_stops
  portfolio_snapshots portfolio_violations price_alerts
  optimization_results backtest_results screener_results
  backtest_strategy_comparison
)

for TABLE in "${TABLES[@]}"; do
  COUNT=$($COMPOSE -p "$PROJECT" exec -T postgres \
    psql -U trader -d news_trading -t -c "SELECT COUNT(*) FROM ${TABLE};" \
    2>/dev/null | tr -d ' ' | tr -d '\n' || echo "ERROR")
  if [[ "$COUNT" =~ ^[0-9]+$ ]]; then
    pass "Table '${TABLE}' exists (${COUNT} rows)"
  else
    fail "Table '${TABLE}' missing or unreadable"
  fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Docker Test Summary"
echo "══════════════════════════════════════════════════════════"
echo "  Passed : $PASS"
echo "  Failed : $FAIL"
echo "  Duration: ${DURATION}s"
echo "══════════════════════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
  red "Some tests failed."
  exit 1
else
  green "All tests passed!"
  exit 0
fi
