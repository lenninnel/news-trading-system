# Deployment Guide

## Architecture

| Service | Purpose | Local port |
|---------|---------|-----------|
| `dashboard` | Streamlit UI | 8501 |
| `health` | FastAPI health-check API | 8080 |
| `worker` | Scheduler (daily runs) | — |
| `monitor` | Price monitor | — |
| `postgres` | PostgreSQL (prod mirror) | 5432 |

---

## Option A — Railway.app (recommended)

### 1. Prerequisites

- [Railway CLI](https://docs.railway.app/develop/cli) installed
- GitHub repository connected to Railway
- A Railway project created

### 2. One-time setup

```bash
# Log in and link your project
railway login
railway link

# Set required environment variables
railway variables set ANTHROPIC_API_KEY=<your-key>
railway variables set NEWSAPI_KEY=<your-key>

# Optional: Telegram alerts
railway variables set TELEGRAM_BOT_TOKEN=<token>
railway variables set TELEGRAM_CHAT_ID=<chat-id>
```

Railway automatically injects `DATABASE_URL` when you attach a PostgreSQL plugin.
Go to **Project → New Service → Database → PostgreSQL** to add one.

### 3. Deploy

```bash
# Git-push triggers auto-deploy (configured in railway.json)
git push origin main

# Or deploy manually with the CLI
./deployment/deploy.sh
```

### 4. Verify

```bash
railway logs                    # tail live logs
railway open                    # open dashboard in browser
curl $(railway domain)/health   # health-check endpoint
```

### Cost (free tier)

| Resource | Limit |
|----------|-------|
| Execution | 500 hrs/month |
| Memory | 512 MB |
| PostgreSQL | 1 GB |
| Bandwidth | 100 GB |

---

## Option B — Render.com

### 1. Connect repository

In the Render dashboard: **New → Blueprint** and point to your repository.
Render reads `render.yaml` and creates all services automatically.

### 2. Set environment variables

In each service's **Environment** tab set:
- `ANTHROPIC_API_KEY`
- `NEWSAPI_KEY`
- `TELEGRAM_BOT_TOKEN` (optional)
- `TELEGRAM_CHAT_ID` (optional)

`DATABASE_URL` is injected automatically from the `news-trading-db` database
defined in `render.yaml`.

### 3. Deploy

Render deploys automatically on every push to `main`.
To trigger manually: **Manual Deploy → Deploy latest commit**.

---

## Local Development

### 1. First-time setup

```bash
# Copy env template and fill in your API keys
./deployment/setup_env.sh
$EDITOR .env

# Start just the dashboard (SQLite backend, no Docker needed)
pip install -r requirements.txt
streamlit run dashboard/app.py
```

### 2. Full local stack (Docker Compose)

```bash
# Start dashboard + PostgreSQL (mirrors production)
docker-compose up

# Also start scheduler and price monitor
docker-compose --profile full up

# Follow logs
docker-compose logs -f
```

Dashboard: http://localhost:8501
Health API: http://localhost:8080/health

---

## SQLite → PostgreSQL Migration

Run **before** switching production traffic to PostgreSQL:

```bash
# 1. Backup SQLite first
./deployment/backup_db.sh

# 2. Dry-run to check row counts
DATABASE_URL=<your-pg-dsn> python3 deployment/migrate_to_postgres.py --dry-run

# 3. Full migration
DATABASE_URL=<your-pg-dsn> python3 deployment/migrate_to_postgres.py
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | ✓ prod | — | Claude API key |
| `NEWSAPI_KEY` | ✓ prod | — | NewsAPI key |
| `DATABASE_URL` | ✓ prod | — | PostgreSQL DSN (`postgresql://...`) |
| `TELEGRAM_BOT_TOKEN` | — | — | Bot token for alerts |
| `TELEGRAM_CHAT_ID` | — | — | Target chat/channel ID |
| `DB_PATH` | — | `news_trading.db` | SQLite file path (local only) |
| `ENVIRONMENT` | — | `development` | `development` or `production` |
| `HEALTH_PORT` | — | `8080` | Port for the FastAPI health server |
| `ACCOUNT_BALANCE` | — | `10000.0` | Paper-trading account size (USD) |

---

## Health Check

The health server exposes:

```
GET /health   — full status JSON (200 healthy, 503 degraded)
GET /ready    — readiness probe for Railway/Render
GET /         — confirm server is running
```

Example response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T09:30:00+00:00",
  "uptime_seconds": 3600,
  "environment": "production",
  "checks": {
    "database": {"ok": true, "message": "ok"},
    "anthropic_key": {"ok": true, "message": "key set"},
    "newsapi_key": {"ok": true, "message": "key set"}
  },
  "last_scheduler_run": "2025-01-15T09:00:00+00:00"
}
```

---

## Rollback

```bash
# List recent Railway deployments
railway deployments

# Roll back to a previous deployment
railway rollback <deployment-id>
```

For Render: use the **Rollback** button in the service's **Deploys** tab.

---

## Database Backups

```bash
# Create a timestamped SQLite backup (local dev)
./deployment/backup_db.sh

# Backups are stored in ./backups/ and the last 30 are kept automatically
```

For PostgreSQL on Railway/Render, use the platform's built-in backup feature
(available on paid plans) or `pg_dump`:

```bash
pg_dump "$DATABASE_URL" -Fc -f "backup_$(date +%Y%m%d).dump"
```
