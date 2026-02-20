#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_tests.sh — Master test runner
#
#  Runs all test suites in order:
#    1. Unit tests        (pytest tests/test_*.py)
#    2. Integration tests (pytest tests/integration_test.py)
#    3. Load test         (python3 tests/load_test_dashboard.py)
#    4. Docker tests      (deployment/test_docker.sh)    — skipped unless --docker
#    5. Migration test    (deployment/test_migration.py) — skipped unless --migrate
#
#  Usage:
#    ./deployment/run_tests.sh                 # unit + integration + load
#    ./deployment/run_tests.sh --all           # all suites including Docker
#    ./deployment/run_tests.sh --unit-only     # unit tests only (fastest)
#    ./deployment/run_tests.sh --docker        # include Docker test (test_docker.sh)
#    ./deployment/run_tests.sh --docker-local  # include local E2E test (test_docker_local.sh)
#    ./deployment/run_tests.sh --migrate       # include migration test
#    ./deployment/run_tests.sh --fast          # unit + integration (skip load)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
RUN_UNIT=true
RUN_INTEGRATION=true
RUN_LOAD=true
RUN_DOCKER=false
RUN_DOCKER_LOCAL=false
RUN_MIGRATE=false

for arg in "$@"; do
  case "$arg" in
    --all)          RUN_DOCKER=true; RUN_DOCKER_LOCAL=true; RUN_MIGRATE=true ;;
    --unit-only)    RUN_INTEGRATION=false; RUN_LOAD=false ;;
    --fast)         RUN_LOAD=false ;;
    --docker)       RUN_DOCKER=true ;;
    --docker-local) RUN_DOCKER_LOCAL=true ;;
    --migrate)      RUN_MIGRATE=true ;;
    --help|-h)
      sed -n '3,20p' "$0" | sed 's/^#  //'
      exit 0 ;;
  esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PASS_SUITES=0
FAIL_SUITES=0
START_TIME=$(date +%s)

# Colour helpers
green()  { echo -e "\033[32m$*\033[0m"; }
red()    { echo -e "\033[31m$*\033[0m"; }
yellow() { echo -e "\033[33m$*\033[0m"; }
header() { echo -e "\n\033[1;34m══════════════════════════════════════════\033[0m"
           echo -e "\033[1;34m  $*\033[0m"
           echo -e "\033[1;34m══════════════════════════════════════════\033[0m"; }

run_suite() {
  local name="$1"
  shift
  header "$name"
  if "$@"; then
    green "✓ $name PASSED"
    ((PASS_SUITES++)) || true
  else
    red   "✗ $name FAILED"
    ((FAIL_SUITES++)) || true
  fi
}

# ── Check Python + pytest ─────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  red "ERROR: python3 not found"
  exit 1
fi
PYTHON="python3"

if ! $PYTHON -m pytest --version &>/dev/null 2>&1; then
  yellow "pytest not found — installing …"
  $PYTHON -m pip install pytest -q
fi

echo ""
green "Python: $($PYTHON --version)"
green "pytest: $($PYTHON -m pytest --version 2>&1 | head -1)"
echo ""

# ── 1. Unit tests ──────────────────────────────────────────────────────────────
if $RUN_UNIT; then
  run_suite "Unit Tests" \
    $PYTHON -m pytest tests/test_coordinator.py tests/test_risk_agent.py \
            tests/test_technical_agent.py \
            -v --tb=short --no-header -q
fi

# ── 2. Integration tests ───────────────────────────────────────────────────────
if $RUN_INTEGRATION; then
  run_suite "Integration Tests" \
    $PYTHON -m pytest tests/integration_test.py \
            -v --tb=short --no-header -q
fi

# ── 3. Load test ───────────────────────────────────────────────────────────────
if $RUN_LOAD; then
  run_suite "Dashboard Load Test" \
    $PYTHON tests/load_test_dashboard.py --users 10 --rows 1000
fi

# ── 4. Docker test (optional) ──────────────────────────────────────────────────
if $RUN_DOCKER; then
  if command -v docker &>/dev/null; then
    run_suite "Docker Integration Test" \
      bash deployment/test_docker.sh
  else
    yellow "Skipping Docker test — Docker not available"
  fi
fi

# ── 4b. Docker local E2E test (optional) ───────────────────────────────────────
if $RUN_DOCKER_LOCAL; then
  if command -v docker &>/dev/null; then
    run_suite "Docker Local E2E Test" \
      bash deployment/test_docker_local.sh
  else
    yellow "Skipping Docker local E2E test — Docker not available"
  fi
fi

# ── 5. Migration test (optional) ───────────────────────────────────────────────
if $RUN_MIGRATE; then
  if [ -f "news_trading.db" ]; then
    run_suite "Migration Test" \
      $PYTHON deployment/test_migration.py
  else
    yellow "Skipping migration test — no news_trading.db found"
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Test Summary"
echo "══════════════════════════════════════════════════════════"
green "  Passed: $PASS_SUITES suite(s)"
[ "$FAIL_SUITES" -gt 0 ] && red "  Failed: $FAIL_SUITES suite(s)" || true
echo "  Duration: ${DURATION}s"
echo "══════════════════════════════════════════════════════════"

if [ "$FAIL_SUITES" -gt 0 ]; then
  red "  OVERALL: FAIL"
  exit 1
else
  green "  OVERALL: PASS"
  exit 0
fi
