# Deployment Guide

## Architecture

| Service | Purpose | Port | Railway service name |
|---------|---------|------|---------------------|
| `web` | Streamlit dashboard | `$PORT` (auto) | `web` |
| `health` | FastAPI `/health` API | `8080` | `health` |
| `worker` | Scheduler daemon (daily runs) | — | `worker` |
| `monitor` | Price alert daemon | — | `monitor` |
| `postgres` | PostgreSQL database | 5432 | Railway plugin |

The **web** and **health** services are HTTP services (publicly reachable).
**worker** and **monitor** are background processes with no public port.

---

## Option A — Railway.app (recommended)

Railway provides the simplest path from code to production.
Free Hobby plan gives $5/month of credits (~170 container-hours).

### Prerequisites

| Tool | Install |
|------|---------|
| Railway CLI | `npm install -g @railway/cli` or `brew install railway` |
| GitHub account | Repository must be public or Railway must have access |

### One-command deploy

```bash
# 1. Copy the env template and fill in your API keys
cp deployment/.env.railway.template .env.railway
$EDITOR .env.railway          # set ANTHROPIC_API_KEY, NEWSAPI_KEY

# 2. Log in to Railway
railway login

# 3. Deploy (creates project, sets vars, deploys dashboard)
./deployment/deploy_railway.sh
```

### Step-by-step manual setup

#### 1. Create a Railway project

Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo** → select your repository.

Railway auto-detects the `Dockerfile` and uses `railway.toml` for the start command.

#### 2. Add a PostgreSQL database

Inside your project:
**+ New Service → Database → PostgreSQL**

Railway automatically injects `DATABASE_URL` into all services in the same project.
The app will migrate from SQLite to PostgreSQL on first startup.

#### 3. Set environment variables

In the Railway dashboard → **web service → Variables → RAW Editor**, paste:

```bash
ANTHROPIC_API_KEY=sk-ant-...
NEWSAPI_KEY=your-newsapi-key
ENVIRONMENT=production
ACCOUNT_BALANCE=10000.0

# Optional — Telegram alerts
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Set AFTER your domain is assigned (Step 5)
DASHBOARD_URL=https://your-app.up.railway.app
```

Or set via CLI:
```bash
railway variables set ANTHROPIC_API_KEY=sk-ant-...
railway variables set NEWSAPI_KEY=your-newsapi-key
railway variables set ENVIRONMENT=production
```

#### 4. Deploy the web service

```bash
# CLI deploy (reads railway.toml for start command)
railway up

# Git push auto-deploys (configured in railway.toml)
git push origin main
```

Railway builds the Docker image, runs the container, and makes it publicly accessible.
First build takes 3–5 minutes. Subsequent deploys take ~60 seconds.

#### 5. Get your public URL

```bash
# Generate a Railway subdomain
railway domain generate

# View assigned domain
railway domain
```

Or in the dashboard: **web service → Settings → Domains → Generate Domain**

Example URL: `https://news-trading-system-production.up.railway.app`

#### 6. Verify the deployment

```bash
# Tail live logs
railway logs

# Open dashboard in browser
railway open

# Health check (should return HTTP 200)
curl https://your-app.up.railway.app/health
```

Expected `/health` response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T09:30:00+00:00",
  "uptime_seconds": 3600,
  "environment": "production",
  "checks": {
    "database":      {"ok": true, "message": "ok"},
    "anthropic_key": {"ok": true, "message": "key set"},
    "newsapi_key":   {"ok": true, "message": "key set"}
  },
  "last_scheduler_run": "2025-01-15T09:00:00+00:00"
}
```

#### 7. Add worker and monitor services (optional)

The web-only deploy runs the dashboard but not the automated scheduler or price alerts.
To add background services:

**In Railway dashboard for each service:**
1. **+ New Service → GitHub Repo** → select same repository
2. **Settings → Deploy → Start Command** → paste command from table below
3. **Variables** → copy all variables from the `web` service
4. Click **Deploy**

| Service | Start command |
|---------|--------------|
| `worker` | `python3 scheduler/daily_runner.py --daemon` |
| `monitor` | `python3 monitoring/price_monitor.py --daemon` |
| `health` | `uvicorn deployment.health_server:app --host 0.0.0.0 --port ${HEALTH_PORT:-8080}` |

> **Cost note:** Each service counts toward your Railway usage hours.
> Worker and monitor are background processes — consider running them only on
> paid plans or during market hours to stay within the free tier.

### Cost estimates (Railway)

| Plan | Monthly credit | Always-on services |
|------|---------------|-------------------|
| Hobby (free) | $5 | ~170 hrs → ~1 service 24/7 |
| Pro | $20 + usage | All 4 services comfortably |

**Per-service cost** (512 MB RAM container):
- ~$0.000231/min = ~$10/month for 24/7 operation

**API costs** (5 tickers × 10 headlines/day × 22 trading days):
- Claude Haiku: ~$0.40/month
- Claude Sonnet: ~$1.49/month

### Troubleshooting Railway

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Build fails | Dockerfile error | `railway logs` → check build output |
| Container crashes on start | Missing env var | Check `ANTHROPIC_API_KEY`, `DATABASE_URL` |
| `/health` returns 503 | DB not connected | Add PostgreSQL plugin; check `DATABASE_URL` |
| Dashboard shows "connecting" | Streamlit cold start | Wait 30–60 s; Railway probes `/_stcore/health` |
| Scheduler not running | `worker` service not created | Add worker service (Step 7) |
| No Telegram alerts | Missing bot token | Set `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` |
| "429 Too Many Requests" | NewsAPI free tier limit | NewsAPI free: 100 req/day; upgrade or add `ALPHA_VANTAGE_KEY` |

```bash
# Most useful debugging commands
railway logs                     # tail all logs
railway logs --service worker    # worker-only logs
railway status                   # show all service status
railway variables                # list all env vars (values hidden)
```

### Rollback

```bash
# List recent deployments
railway deployments

# Roll back to a specific deployment
railway rollback <deployment-id>
```

In the dashboard: **web service → Deployments → ⋯ → Rollback**

---

## Option B — Render.com

Render is an alternative PaaS with a generous free tier for web services.

### 1. Connect repository

Dashboard → **New → Blueprint** → select your repository.
Render reads `render.yaml` and creates all services automatically.

### 2. Set environment variables

In each service's **Environment** tab set:
- `ANTHROPIC_API_KEY`
- `NEWSAPI_KEY`
- `TELEGRAM_BOT_TOKEN` (optional)
- `TELEGRAM_CHAT_ID` (optional)

`DATABASE_URL` is injected automatically from the PostgreSQL instance defined in `render.yaml`.

### 3. Deploy

Render deploys automatically on every push to `main`.
To trigger manually: **Manual Deploy → Deploy latest commit**.

---

## Local Development

### Option 1 — No Docker (fastest)

```bash
# Install dependencies
pip install -r requirements.txt

# Copy env template
cp .env.docker .env
$EDITOR .env     # fill in ANTHROPIC_API_KEY, NEWSAPI_KEY

# Start the dashboard (SQLite backend, no PostgreSQL needed)
streamlit run dashboard/app.py

# Run a single trading cycle
python3 scheduler/daily_runner.py --now

# Start the price monitor (separate terminal)
python3 monitoring/price_monitor.py --check-now
```

### Option 2 — Docker Compose (mirrors production)

```bash
# Copy env template
cp .env.docker .env
$EDITOR .env

# Start dashboard + PostgreSQL
docker compose up

# Also start scheduler worker and price monitor
docker compose --profile full up

# Follow logs
docker compose logs -f
```

| URL | Service |
|-----|---------|
| http://localhost:8501 | Streamlit dashboard |
| http://localhost:8080/health | Health check API |
| localhost:5432 | PostgreSQL (psql, TablePlus, etc.) |

### Option 3 — Full local E2E test

```bash
# Run the complete Docker stack validation (~5 min)
./deployment/test_docker_local.sh

# With real API keys (runs a full scheduler cycle):
FULL_TEST=1 ./deployment/test_docker_local.sh
```

Expected output:
```
✓ PostgreSQL ready
✓ Migrations applied (17 tables)
✓ Health check passed
✓ Scheduler run successful
✓ Dashboard accessible
```

---

## SQLite → PostgreSQL Migration

Run **before** switching production traffic to a new PostgreSQL instance:

```bash
# 1. Backup SQLite first
./deployment/backup_db.sh

# 2. Dry-run — shows row counts, makes no changes
DATABASE_URL=<your-pg-dsn> python3 deployment/migrate_to_postgres.py --dry-run

# 3. Full migration (truncates PG tables, copies all data from SQLite)
DATABASE_URL=<your-pg-dsn> python3 deployment/migrate_to_postgres.py
```

The migration script is idempotent — safe to run multiple times.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | ✓ prod | — | Claude API key (`sk-ant-...`) |
| `NEWSAPI_KEY` | ✓ prod | — | NewsAPI.org key |
| `DATABASE_URL` | ✓ prod | — | PostgreSQL DSN (`postgresql://...`). Injected by Railway. |
| `ALPHA_VANTAGE_KEY` | — | — | Fallback price source (free at alphavantage.co) |
| `TELEGRAM_BOT_TOKEN` | — | — | Telegram bot token for trade alerts |
| `TELEGRAM_CHAT_ID` | — | — | Telegram chat/channel ID |
| `DASHBOARD_URL` | — | `http://localhost:8501` | Shown as link in Telegram messages |
| `DB_PATH` | — | `news_trading.db` | SQLite path (local only; overridden by `DATABASE_URL`) |
| `ENVIRONMENT` | — | `development` | `development` or `production` |
| `HEALTH_PORT` | — | `8080` | Port for the FastAPI health server (health service only) |
| `ACCOUNT_BALANCE` | — | `10000.0` | Paper-trading account size (USD) |

---

## Health Check Endpoints

```
GET /health   → 200 (healthy) or 503 (degraded) — full status JSON
GET /ready    → 200 (ready) or 503 (not ready) — readiness probe
GET /         → 200 always — confirms server is running
```

The health server runs as a **separate Railway service** (`health`) on port 8080.
It is independent of the Streamlit dashboard.

---

## Database Backups

```bash
# Local SQLite backup (timestamped, keeps last 30)
./deployment/backup_db.sh

# PostgreSQL dump (Railway / any remote)
pg_dump "$DATABASE_URL" -Fc -f "backup_$(date +%Y%m%d_%H%M).dump"

# Restore PostgreSQL
pg_restore -d "$DATABASE_URL" backup_20250115_0900.dump
```

For automated backups on Railway (paid plans): use the built-in
**Backups** feature in the PostgreSQL plugin settings.
