# Cost Monitoring Guide — News Trading System

> **Audience**: Operators who want to stay within free-tier limits or understand the cost of scaling.

---

## Table of Contents

1. [Railway.app Cost Breakdown](#1-railwayapp-cost-breakdown)
2. [API Costs](#2-api-costs)
3. [Staying in the Free Tier](#3-staying-in-the-free-tier)
4. [Cost Monitoring Dashboard](#4-cost-monitoring-dashboard)
5. [Cost Scaling Reference](#5-cost-scaling-reference)

---

## 1. Railway.app Cost Breakdown

### Free tier limits (Hobby trial / Starter)

| Resource | Free Allowance | Typical Usage (5 tickers) |
|---|---|---|
| Compute hours | 500 hrs/month | ~720 hrs/month (3 services × 24/7) |
| RAM per service | 512 MB | ~200–350 MB per service |
| Disk | 1 GB | ~50–200 MB (SQLite DB) |
| Outbound bandwidth | 100 GB | < 1 GB/month |
| PostgreSQL (plugin) | $0 for first DB | $0 |

**Key insight:** 500 compute hours = ~20 days of a single always-on service. Running 3 services (dashboard + worker + monitor) 24/7 = 2,160 hrs/month — **well over the free tier**.

### When you'll hit the free tier

| Scenario | Time to hit limit |
|---|---|
| 1 service (worker only, no dashboard) | ~20 days |
| 3 services always-on | ~7 days |
| Scheduler only (runs 5 min/day) | Never — use sleep mode |

### Recommended configuration to minimise cost

```
dashboard  → deploy only when needed (scale to zero when not in use)
worker     → runs 5–15 min/day (scheduler) — Railway sleeps it otherwise
monitor    → disable if not using price alerts
postgres   → included free, no extra charge
```

### Estimated monthly cost by plan

| Plan | Price | What you get | Suitable for |
|---|---|---|---|
| **Trial** | $0 | 500 compute hrs total (one-time) | Testing |
| **Hobby** | $5/month | $5 credit + usage-based after | Single-service light use |
| **Pro** | $20/month | $20 credit + usage-based | 3–5 services always-on |
| **Team** | $100/month | $100 credit + shared team | Production multi-user |

**Realistic monthly cost at scale (3 services, weekdays only):**
- Worker: 22 days × 15 min/day = ~5.5 compute hrs → ~$0.10
- Dashboard: 22 days × 8 hrs/day = ~176 compute hrs → ~$3.50
- Monitor: 22 days × 24 hrs/day = ~528 compute hrs → ~$10.50
- PostgreSQL: $0 (included)
- **Total: ~$14–15/month** on the Pro plan ($20 credit covers it)

### Railway-specific cost tips

- Use **sleep mode** for the dashboard (`railway service sleep`) when not actively viewing it.
- Set `RAILWAY_DEPLOYMENT_OVERLAP_SECONDS=0` to avoid double-billing during deploys.
- Monitor usage at [railway.app/dashboard](https://railway.app/dashboard) → Usage tab.

---

## 2. API Costs

### NewsAPI

| Tier | Price | Requests | Headlines/req | Total headlines/day |
|---|---|---|---|---|
| **Developer (free)** | $0 | 100 req/day | up to 10 | 1,000 |
| **Business** | $449/month | 250,000 req/month | up to 100 | unlimited |

**Typical usage (5 tickers, 1 run/day):**
- 5 tickers × 1 req/ticker = **5 requests/day**
- Free tier limit: 100 req/day → **safe for up to 20 tickers/day**
- Screener uses additional requests (1 req per screener batch call)

**When you'll hit the free tier:**
- Running the screener across all markets (400+ tickers): can use 50–100 req in one scan
- Running the scheduler more than once per day with a large watchlist
- NewsAPI free tier also **requires attribution** and blocks commercial use

**Upgrade trigger:** If `recovery_log` shows `newsapi` events with `event_type = 'rate_limited'` more than 3×/week.

---

### Anthropic (Claude API)

| Model | Input price | Output price | Typical tokens/headline |
|---|---|---|---|
| claude-haiku-4-5 | $0.80/M tokens | $4.00/M tokens | ~200 in / ~50 out |
| claude-sonnet-4-6 | $3.00/M tokens | $15.00/M tokens | ~200 in / ~50 out |

**Default model:** `claude-sonnet-4-6` (configurable in `config/settings.py`)

**Cost per headline analysis:**

| Model | Input cost | Output cost | Per headline |
|---|---|---|---|
| claude-haiku-4-5 | 200 × $0.80/1M = $0.00016 | 50 × $4.00/1M = $0.0002 | **~$0.00036** |
| claude-sonnet-4-6 | 200 × $3.00/1M = $0.0006 | 50 × $15.00/1M = $0.00075 | **~$0.00135** |

**Daily cost estimate (5 tickers × 10 headlines):**

| Model | Daily cost | Monthly cost (22 trading days) |
|---|---|---|
| claude-haiku-4-5 | $0.018 | **$0.40** |
| claude-sonnet-4-6 | $0.068 | **$1.49** |

**To switch to Haiku (reduce cost by ~4×):**
```python
# config/settings.py
CLAUDE_MODEL = "claude-haiku-4-5-20251001"   # was: claude-sonnet-4-6
```

**Free credits:** Anthropic gives $5–$10 in free credits to new accounts. At Sonnet pricing this covers ~3,700 headline analyses — roughly 74 trading days at 5 tickers × 10 headlines.

**Upgrade trigger:** Monthly spend > $5 (check at console.anthropic.com).

---

### Alpha Vantage (Price Fallback — Level 1)

| Tier | Price | Requests | Notes |
|---|---|---|---|
| **Free** | $0 | 25 req/day | 5 req/min rate limit |
| **Premium** | $50/month | 75 req/min | No daily cap |

**Usage in this system:** Alpha Vantage is Level 1 price fallback — only used when yfinance fails. Normal usage = **0 requests/day**. Only billed when yfinance is degraded.

**When you'll hit the free tier:** If yfinance is broken for an entire day AND you have > 25 tickers in your watchlist.

**Setup:** Add `ALPHA_VANTAGE_KEY` to `.env`. Leave blank to skip this fallback level.

---

### yfinance (Primary Price Source)

**Cost: $0** — yfinance scrapes Yahoo Finance and has no API key or billing. Rate limits are soft and handled automatically by the circuit breaker.

---

### Google News RSS (News Fallback — Level 2)

**Cost: $0** — public RSS endpoint, no authentication required.

---

## 3. Staying in the Free Tier

### Configuration checklist

```yaml
# config/watchlist.yaml — tune these to minimise API calls

watchlist:
  - AAPL      # ≤ 10 tickers to stay safely within NewsAPI free tier
  - NVDA
  - TSLA

scheduler:
  weekdays_only: true  # saves ~30% by skipping weekends

# config/settings.py
MAX_HEADLINES = 5      # reduce from 10 → 5 to halve Anthropic costs
CLAUDE_MODEL  = "claude-haiku-4-5-20251001"  # 4× cheaper than Sonnet
```

### Free-tier budget by component (5 tickers, Haiku model)

| Component | Free allowance | Est. usage | Status |
|---|---|---|---|
| NewsAPI requests | 100/day | 5–10/day | ✅ Safe |
| Anthropic tokens | $5 credit | $0.40/month | ✅ ~12 months free |
| Alpha Vantage | 25/day | 0/day (fallback only) | ✅ Safe |
| Railway compute | 500 hrs trial | 5–10 hrs/month (worker only) | ✅ Safe for months |
| yfinance / Google RSS | Unlimited | — | ✅ Always free |

### Tips to minimise costs

1. **Use Haiku instead of Sonnet** — 4× cheaper, negligible quality difference for financial headlines.
2. **Reduce `MAX_HEADLINES`** — 5 headlines/ticker captures 90% of the sentiment signal.
3. **Don't run screener daily** — run it weekly or on demand; it's the biggest NewsAPI consumer.
4. **Sleep Railway services** — put the dashboard to sleep when you're not watching it.
5. **Use the fallback chain** — if yfinance is working, you never touch Alpha Vantage.
6. **Cache aggressively** — headlines cached for 24h mean repeated runs of the same ticker cost nothing extra.
7. **Weekdays only** — `weekdays_only: true` eliminates 2/7 of all API calls.

---

## 4. Cost Monitoring Dashboard

### Check current API usage

```bash
# NewsAPI calls today (count from recovery_log)
sqlite3 news_trading.db \
  "SELECT count(*) AS newsapi_calls_today
   FROM recovery_log
   WHERE service = 'newsapi'
     AND event_type = 'api_call'
     AND date(created_at) = date('now');"

# Anthropic calls this month and estimated cost
sqlite3 news_trading.db \
  "SELECT
     count(*) AS claude_calls,
     count(*) * 250 AS est_input_tokens,
     count(*) * 50  AS est_output_tokens,
     round(count(*) * 0.00135, 4) AS est_cost_usd_sonnet,
     round(count(*) * 0.00036, 4) AS est_cost_usd_haiku
   FROM recovery_log
   WHERE service = 'anthropic'
     AND event_type = 'api_call'
     AND strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now');"

# Fallback activations this month (indicates primary source issues)
sqlite3 news_trading.db \
  "SELECT service, event_type, count(*) AS n
   FROM recovery_log
   WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')
   GROUP BY service, event_type
   ORDER BY n DESC;"
```

### Monthly cost report script

```bash
#!/bin/bash
# Save as scripts/cost_report.sh and run monthly

DB="news_trading.db"
MONTH=$(date +%Y-%m)

echo "=== Cost Report: $MONTH ==="
echo ""
echo "--- API Call Counts ---"
sqlite3 "$DB" \
  "SELECT service, count(*) AS calls
   FROM recovery_log
   WHERE strftime('%Y-%m', created_at) = '$MONTH'
     AND event_type = 'api_call'
   GROUP BY service ORDER BY calls DESC;"

echo ""
echo "--- Estimated Anthropic Cost (Sonnet) ---"
sqlite3 "$DB" \
  "SELECT '$' || round(count(*) * 0.00135, 2) AS est_usd
   FROM recovery_log
   WHERE service='anthropic' AND event_type='api_call'
     AND strftime('%Y-%m', created_at) = '$MONTH';"

echo ""
echo "--- Fallback Activations (higher = more primary failures) ---"
sqlite3 "$DB" \
  "SELECT service, event_type, count(*) AS n
   FROM recovery_log
   WHERE strftime('%Y-%m', created_at) = '$MONTH'
     AND event_type NOT IN ('api_call', 'circuit_closed')
   GROUP BY service, event_type ORDER BY n DESC;"
```

---

## 5. Cost Scaling Reference

Expected monthly costs as the system scales:

| Watchlist size | Model | NewsAPI plan | Anthropic/month | Railway/month | Total/month |
|---|---|---|---|---|---|
| 5 tickers | Haiku | Free | ~$0.40 | $0 (trial) | **~$0.40** |
| 10 tickers | Haiku | Free | ~$0.80 | ~$5 (Hobby) | **~$5.80** |
| 20 tickers | Haiku | Free (tight) | ~$1.60 | ~$5 (Hobby) | **~$6.60** |
| 20 tickers | Sonnet | Free (tight) | ~$6.00 | ~$5 (Hobby) | **~$11.00** |
| 50 tickers | Sonnet | Business ($449) | ~$15.00 | ~$20 (Pro) | **~$484/month** |

> At 50 tickers, the NewsAPI Business plan ($449/month) dominates total cost. At this scale, consider caching headlines aggressively and running analysis only on tickers where the screener detects significant volume/price movement.
