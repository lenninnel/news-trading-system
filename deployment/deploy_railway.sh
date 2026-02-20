#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  deploy_railway.sh — Deploy the News Trading System to Railway.app
#
#  What this script does:
#    1. Validates prerequisites (Railway CLI, git, curl)
#    2. Checks Railway authentication
#    3. Loads environment from .env.railway or .env
#    4. Validates required API keys
#    5. Links or initialises a Railway project
#    6. Provisions a PostgreSQL database (if not already present)
#    7. Sets all environment variables on the web service
#    8. Deploys the web (Streamlit dashboard) service
#    9. Generates a public domain (if needed)
#   10. Sets up worker, monitor, and health services (with instructions)
#   11. Shows live deployment status and URLs
#
#  Usage:
#    ./deployment/deploy_railway.sh              # full deploy
#    ./deployment/deploy_railway.sh --web-only   # deploy web only (fast)
#    ./deployment/deploy_railway.sh --env-only   # set env vars, skip deploy
#    ./deployment/deploy_railway.sh --status     # show deployment status only
#    ./deployment/deploy_railway.sh --logs       # tail live logs
#
#  Prerequisites:
#    npm install -g @railway/cli    (or brew install railway)
#    railway login
#    Fill in deployment/.env.railway.template → .env.railway
#
#  Target: first deploy ~5 min (Docker build), re-deploys ~60 s
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Parse flags ───────────────────────────────────────────────────────────────
WEB_ONLY=false
ENV_ONLY=false
STATUS_ONLY=false
SHOW_LOGS=false

for arg in "${@:-}"; do
  case "$arg" in
    --web-only)  WEB_ONLY=true ;;
    --env-only)  ENV_ONLY=true ;;
    --status)    STATUS_ONLY=true ;;
    --logs)      SHOW_LOGS=true ;;
    --help|-h)
      sed -n '3,30p' "$0" | sed 's/^#  //'
      exit 0 ;;
  esac
done

# ── Change to project root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Colour helpers ────────────────────────────────────────────────────────────
green()  { printf "\033[32m✓ %s\033[0m\n" "$*"; }
red()    { printf "\033[31m✗ %s\033[0m\n" "$*"; }
yellow() { printf "\033[33m• %s\033[0m\n" "$*"; }
cyan()   { printf "\033[36m── %s ──\033[0m\n" "$*"; }
bold()   { printf "\033[1m%s\033[0m\n" "$*"; }
info()   { printf "  %s\n" "$*"; }

START_TIME=$(date +%s)

# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Status / Logs shortcuts
# ══════════════════════════════════════════════════════════════════════════════

if $STATUS_ONLY; then
  cyan "Deployment Status"
  railway status 2>/dev/null || { red "Not linked to a Railway project (run railway link)"; exit 1; }
  echo ""
  yellow "Services:"
  railway service 2>/dev/null || true
  echo ""
  yellow "Domain:"
  railway domain 2>/dev/null || true
  exit 0
fi

if $SHOW_LOGS; then
  yellow "Streaming Railway logs (Ctrl+C to stop) …"
  railway logs --tail 2>/dev/null
  exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Prerequisites
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 1 — Prerequisites"

if ! command -v railway &>/dev/null; then
  red "Railway CLI not found"
  info "Install with: npm install -g @railway/cli"
  info "             OR: brew install railway"
  info "Then run:    railway login"
  exit 1
fi
RAILWAY_VER=$(railway --version 2>/dev/null | head -1 || echo "unknown")
green "Railway CLI: $RAILWAY_VER"

if ! command -v git &>/dev/null; then
  red "git not found"
  exit 1
fi
green "git: $(git --version | head -1)"

if ! command -v curl &>/dev/null; then
  red "curl not found"
  exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Authentication
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 2 — Railway authentication"

RAILWAY_USER=$(railway whoami 2>/dev/null || echo "")
if [ -z "$RAILWAY_USER" ]; then
  red "Not authenticated with Railway"
  info "Run: railway login"
  exit 1
fi
green "Authenticated as: $RAILWAY_USER"

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Load environment variables
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 3 — Load environment"

ENV_FILE=""
if [ -f ".env.railway" ]; then
  ENV_FILE=".env.railway"
elif [ -f ".env" ]; then
  ENV_FILE=".env"
fi

if [ -n "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
  yellow "Loaded from: $ENV_FILE"
else
  yellow "No .env.railway or .env found — using current shell environment"
  yellow "Tip: cp deployment/.env.railway.template .env.railway"
  yellow "     (fill in your API keys, then re-run this script)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Validate required variables
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 4 — Validate configuration"

MISSING_VARS=()
WARN_VARS=()

# Required for full functionality
for var in ANTHROPIC_API_KEY NEWSAPI_KEY; do
  val="${!var:-}"
  if [ -z "$val" ] || [[ "$val" == *"..."* ]] || [[ "$val" == *"your_"* ]]; then
    MISSING_VARS+=("$var")
  else
    green "$var is set"
  fi
done

# Optional but recommended
for var in TELEGRAM_BOT_TOKEN DASHBOARD_URL; do
  val="${!var:-}"
  if [ -z "$val" ]; then
    WARN_VARS+=("$var")
  fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
  red "Required variables not set: ${MISSING_VARS[*]}"
  info "Edit .env.railway and set real values, then re-run."
  info "Get API keys:"
  info "  ANTHROPIC_API_KEY → https://console.anthropic.com/"
  info "  NEWSAPI_KEY       → https://newsapi.org/register"
  exit 1
fi

if [ ${#WARN_VARS[@]} -gt 0 ]; then
  yellow "Optional variables not set (non-fatal): ${WARN_VARS[*]}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Project setup
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 5 — Railway project"

# Check if already linked
if railway status &>/dev/null 2>&1; then
  green "Already linked to a Railway project"
  railway status 2>/dev/null | head -5 || true
else
  echo ""
  bold "No Railway project linked. Choose an option:"
  echo "  1) Link to an existing project:  railway link"
  echo "  2) Create a new project:         railway init"
  echo ""
  read -rp "Run which command? [link/init/skip]: " CHOICE
  case "$CHOICE" in
    link)  railway link ;;
    init)  railway init ;;
    skip)  yellow "Skipping — you can link manually later" ;;
    *)     yellow "Skipping" ;;
  esac
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — PostgreSQL database
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 6 — PostgreSQL database"

# Check if DATABASE_URL is already set in Railway (injected by plugin)
DB_URL_SET=$(railway variables get DATABASE_URL 2>/dev/null || echo "")
if [ -n "$DB_URL_SET" ]; then
  green "PostgreSQL already attached (DATABASE_URL is set)"
else
  yellow "DATABASE_URL not found in Railway variables"
  info "Add PostgreSQL to your project:"
  info "  → Railway Dashboard → your project → New Service → Database → PostgreSQL"
  info "  OR via CLI: railway add --plugin postgresql"
  info ""
  # Try CLI addition
  if railway add --plugin postgresql &>/dev/null 2>&1; then
    green "PostgreSQL plugin added via CLI"
  else
    yellow "Could not add PostgreSQL via CLI — please add manually in the dashboard"
    yellow "The app will use SQLite as fallback until DATABASE_URL is set."
  fi
fi

if $ENV_ONLY; then
  # Skip to env var setting
  true
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Set environment variables on the web service
# ══════════════════════════════════════════════════════════════════════════════
# Helper: set one Railway variable; silently skips empty values.
# Must be defined before Step 7 and Step 8 both use it.
SET_COUNT=0
SKIP_COUNT=0

_railway_set_var() {
  local key="$1"
  local val="$2"
  local svc_flag="${3:-}"   # optional: "--service NAME"
  if [ -z "$val" ]; then
    ((SKIP_COUNT++)) || true
    return
  fi
  # shellcheck disable=SC2086
  if railway variables set "${key}=${val}" ${svc_flag} &>/dev/null 2>&1; then
    ((SET_COUNT++)) || true
  else
    yellow "  Could not set ${key} — add it in the Railway dashboard → Variables"
  fi
}

_set_all_vars() {
  local svc_flag="${1:-}"
  set +u
  _railway_set_var "ANTHROPIC_API_KEY" "${ANTHROPIC_API_KEY:-}"  "$svc_flag"
  _railway_set_var "NEWSAPI_KEY"        "${NEWSAPI_KEY:-}"        "$svc_flag"
  _railway_set_var "ALPHA_VANTAGE_KEY"  "${ALPHA_VANTAGE_KEY:-}"  "$svc_flag"
  _railway_set_var "TELEGRAM_BOT_TOKEN" "${TELEGRAM_BOT_TOKEN:-}" "$svc_flag"
  _railway_set_var "TELEGRAM_CHAT_ID"   "${TELEGRAM_CHAT_ID:-}"   "$svc_flag"
  _railway_set_var "ENVIRONMENT"        "${ENVIRONMENT:-production}" "$svc_flag"
  _railway_set_var "HEALTH_PORT"        "${HEALTH_PORT:-8080}"    "$svc_flag"
  _railway_set_var "ACCOUNT_BALANCE"    "${ACCOUNT_BALANCE:-10000.0}" "$svc_flag"
  _railway_set_var "DASHBOARD_URL"      "${DASHBOARD_URL:-}"      "$svc_flag"
  set -u
}

# ── Step 7 preview (actual setting happens after deploy creates the service) ──
cyan "Step 7 — Environment variables (set after deploy)"
yellow "Variables will be pushed to Railway after the service is created in Step 8."
yellow "Preparing values from: ${ENV_FILE:-shell environment}"

if $ENV_ONLY; then
  # In env-only mode there may already be a service — try immediately
  SVC_NAME=$(railway status 2>/dev/null | grep -oE 'Service: .+' | sed 's/Service: //' | tr -d '[:space:]' || echo "")
  SVC_FLAG=""
  [ -n "$SVC_NAME" ] && [ "$SVC_NAME" != "None" ] && SVC_FLAG="--service ${SVC_NAME}"
  _set_all_vars "$SVC_FLAG"
  green "Environment variables set: $SET_COUNT (${SKIP_COUNT} empty/skipped)"
  if [ "$SET_COUNT" -eq 0 ]; then
    _print_dashboard_vars
  fi
  green "ENV_ONLY mode — done."
  exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Deploy web service (Streamlit dashboard)
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 8 — Deploy web service"

yellow "Building and deploying (first build ~3-5 min, subsequent ~60 s) …"
yellow "Tip: tail logs in another terminal with: railway logs"
echo ""

if railway up --detach 2>&1; then
  green "Deployment submitted to Railway"
else
  red "Deployment failed — check: railway logs"
  exit 1
fi

# ── Now set env vars (service exists after railway up) ────────────────────────
cyan "Step 7b — Push environment variables to service"
sleep 2   # give Railway a moment to register the service

# Discover the deployed service name from `railway service status --all`
SVC_NAME=$(railway service status --all 2>/dev/null \
  | grep -oE '^[a-zA-Z0-9_-]+' | head -1 || echo "")
SVC_FLAG=""
if [ -n "$SVC_NAME" ] && [ "$SVC_NAME" != "Services" ]; then
  # Link the CLI context to this service so future commands target it
  railway service link "$SVC_NAME" &>/dev/null 2>&1 || true
  SVC_FLAG="--service ${SVC_NAME}"
  yellow "Targeting service: $SVC_NAME"
fi

_set_all_vars "$SVC_FLAG"
green "Environment variables set: $SET_COUNT (${SKIP_COUNT} empty/skipped)"

# If CLI couldn't set vars, print them for manual dashboard entry
if [ "$SET_COUNT" -eq 0 ]; then
  echo ""
  bold "⚠ Paste these into Railway Dashboard → your service → Variables → RAW Editor:"
  echo "──────────────────────────────────────────────────────────"
  set +u
  echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}"
  echo "NEWSAPI_KEY=${NEWSAPI_KEY:-}"
  [ -n "${ALPHA_VANTAGE_KEY:-}"  ] && echo "ALPHA_VANTAGE_KEY=${ALPHA_VANTAGE_KEY:-}"
  [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && echo "TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-}"
  [ -n "${TELEGRAM_CHAT_ID:-}"   ] && echo "TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID:-}"
  set -u
  echo "ENVIRONMENT=production"
  echo "HEALTH_PORT=8080"
  echo "ACCOUNT_BALANCE=${ACCOUNT_BALANCE:-10000.0}"
  echo "──────────────────────────────────────────────────────────"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — Generate or show domain
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 9 — Public domain"

# Wait a moment for the deployment to register
sleep 3

DOMAIN=$(railway domain 2>/dev/null | grep -oE 'https?://[^[:space:]]+' | head -1 || echo "")
if [ -n "$DOMAIN" ]; then
  green "Dashboard URL: $DOMAIN"
else
  # Try to generate a domain (railway domain with no args generates one)
  if railway domain &>/dev/null 2>&1; then
    DOMAIN=$(railway domain 2>/dev/null | grep -oE 'https?://[^[:space:]]+' | head -1 || echo "")
    if [ -n "$DOMAIN" ]; then
      green "Dashboard URL: $DOMAIN"
    fi
  fi
  if [ -z "$DOMAIN" ]; then
    yellow "Domain not yet assigned — run 'railway domain generate' or check the dashboard"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 10 — Worker, monitor, and health services
# ══════════════════════════════════════════════════════════════════════════════
if ! $WEB_ONLY; then
  cyan "Step 10 — Additional services"
  echo ""
  bold "The web (dashboard) service is being deployed."
  bold "To run the full system, add these services in the Railway dashboard:"
  echo ""

  printf "  \033[1m%-12s\033[0m  %s\n" "Service" "Start command"
  printf "  %-12s  %s\n" "──────────" "──────────────────────────────────────────────────────"
  printf "  \033[36m%-12s\033[0m  %s\n" "worker" \
    "python3 scheduler/daily_runner.py --daemon"
  printf "  \033[36m%-12s\033[0m  %s\n" "monitor" \
    "python3 monitoring/price_monitor.py --daemon"
  printf "  \033[36m%-12s\033[0m  %s\n" "health" \
    "uvicorn deployment.health_server:app --host 0.0.0.0 --port \${HEALTH_PORT:-8080}"
  echo ""

  bold "Steps to add each service:"
  info "1. Railway Dashboard → your project → + New Service → GitHub Repo"
  info "2. Select this repository"
  info "3. In the service settings → Deploy → Start command → paste from above"
  info "4. In Variables → copy all variables from the web service"
  info "5. Click Deploy"
  echo ""

  # Try adding worker/monitor via CLI
  for SVC_NAME in worker monitor health; do
    case "$SVC_NAME" in
      worker)  SVC_CMD="python3 scheduler/daily_runner.py --daemon" ;;
      monitor) SVC_CMD="python3 monitoring/price_monitor.py --daemon" ;;
      health)  SVC_CMD="uvicorn deployment.health_server:app --host 0.0.0.0 --port \${HEALTH_PORT:-8080}" ;;
    esac

    # Attempt to create service via CLI (may not be supported in all CLI versions)
    if railway service create "$SVC_NAME" &>/dev/null 2>&1; then
      # Set start command as an env var that the container reads (or redeploy with command)
      railway variables set "RAILWAY_RUN_COMMAND=${SVC_CMD}" --service "$SVC_NAME" &>/dev/null 2>&1 || true

      # Copy API keys to new service (set +u guards indirect expansion)
      set +u
      for var in ANTHROPIC_API_KEY NEWSAPI_KEY ALPHA_VANTAGE_KEY TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID ENVIRONMENT ACCOUNT_BALANCE; do
        val="${!var:-}"
        [ -n "$val" ] && railway variables set "${var}=${val}" --service "$SVC_NAME" &>/dev/null 2>&1 || true
      done
      set -u
      green "Service '$SVC_NAME' created (set start command in dashboard)"
    fi
  done
fi

# ══════════════════════════════════════════════════════════════════════════════
# Step 11 — Deployment status
# ══════════════════════════════════════════════════════════════════════════════
cyan "Step 11 — Status"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
bold "══════════════════════════════════════════════════════════"
bold "  Railway Deployment Summary"
bold "══════════════════════════════════════════════════════════"
[ -n "$DOMAIN" ] && printf "  %-18s %s\n" "Dashboard URL:" "$DOMAIN"
printf "  %-18s %s\n" "Health check:" "${DOMAIN:-<pending>}/health"
printf "  %-18s %s\n" "Duration:" "${DURATION}s"
bold "══════════════════════════════════════════════════════════"
echo ""
bold "Next steps:"
info "• Tail logs:          railway logs"
info "• Open dashboard:     railway open"
info "• Check health:       curl ${DOMAIN:-<your-domain>}/health"
info "• Set Telegram vars:  railway variables set TELEGRAM_BOT_TOKEN=..."
info "• Update DASHBOARD_URL after domain assigned:"
info "    railway variables set DASHBOARD_URL=${DOMAIN:-https://your-app.up.railway.app}"
echo ""
if ! $WEB_ONLY; then
  bold "Add worker + monitor services via Railway dashboard for:"
  info "• Automated daily scheduling (market open: 09:30 ET)"
  info "• Real-time price alerts and stop-loss monitoring"
  info "• Telegram trade notifications"
fi
echo ""
green "Deployment complete — web service building on Railway"
