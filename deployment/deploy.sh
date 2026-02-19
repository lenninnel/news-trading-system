#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  deploy.sh — One-command deployment helper
#
#  Targets: Railway CLI or Render (git-push based)
#
#  Usage:
#    ./deployment/deploy.sh              # deploy current branch to Railway
#    ./deployment/deploy.sh --render     # push to git remote (triggers Render)
#    ./deployment/deploy.sh --local      # build + run locally via Docker Compose
#    ./deployment/deploy.sh --migrate    # run SQLite → PostgreSQL migration only
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODE="${1:-}"

# ── Helpers ──────────────────────────────────────────────────────────────────

require_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "ERROR: '$1' is not installed or not on PATH."
    exit 1
  fi
}

check_env() {
  local missing=()
  for var in ANTHROPIC_API_KEY NEWSAPI_KEY; do
    [ -z "${!var:-}" ] && missing+=("$var")
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "WARNING: the following env vars are not set: ${missing[*]}"
    echo "         The app will start but may not function correctly."
  fi
}

# ── Local Docker ──────────────────────────────────────────────────────────────

if [ "$MODE" = "--local" ]; then
  require_cmd docker
  require_cmd docker-compose
  echo "Building and starting local stack …"
  docker-compose build
  docker-compose up -d postgres
  echo "Waiting for PostgreSQL to be healthy …"
  sleep 5
  docker-compose up -d dashboard health
  echo ""
  echo "Dashboard : http://localhost:8501"
  echo "Health API: http://localhost:8080/health"
  echo ""
  echo "Tip: docker-compose --profile full up -d  (start scheduler + monitor too)"
  exit 0
fi

# ── Migration only ────────────────────────────────────────────────────────────

if [ "$MODE" = "--migrate" ]; then
  if [ -z "${DATABASE_URL:-}" ]; then
    echo "ERROR: DATABASE_URL must be set for migration."
    exit 1
  fi
  echo "Running SQLite → PostgreSQL migration …"
  python3 deployment/migrate_to_postgres.py
  exit 0
fi

# ── Render (git-push) ────────────────────────────────────────────────────────

if [ "$MODE" = "--render" ]; then
  require_cmd git
  echo "Pushing to git remote (Render auto-deploys on push) …"
  git push origin main
  echo "Deployment triggered. Monitor at https://dashboard.render.com"
  exit 0
fi

# ── Railway CLI (default) ─────────────────────────────────────────────────────

require_cmd railway
check_env

echo "Deploying to Railway …"
railway up

echo ""
echo "Deployment submitted. Check progress with: railway logs"
echo "Open dashboard with:                        railway open"
