# Operations Guide — News Trading System

> **Audience**: Operators running the system daily, monitoring performance, and managing risk.

---

## Table of Contents

1. [Daily Operations Checklist](#1-daily-operations-checklist)
2. [Running the Scheduler](#2-running-the-scheduler)
3. [Monitoring Performance](#3-monitoring-performance)
4. [Interpreting Signals](#4-interpreting-signals)
5. [Risk Management Guidelines](#5-risk-management-guidelines)
6. [Database Maintenance](#6-database-maintenance)
7. [Log Management](#7-log-management)

---

## 1. Daily Operations Checklist

### Pre-market (before market open)

- [ ] Confirm the scheduler ran (check `scheduler/logs/YYYY-MM-DD.log`)
- [ ] Review any Telegram alerts received overnight
- [ ] Open the dashboard at [http://localhost:8501](http://localhost:8501)
- [ ] Check the **Overview** page for signals generated today

### At market open (09:30 ET for US / 09:00 CET for European)

- [ ] If scheduler has not run yet: `python3 scheduler/daily_runner.py --now`
- [ ] Run the stock screener to find high-momentum candidates: `python3 -m agents.screener_agent --markets US DE EU --top 20`
- [ ] Review STRONG BUY and STRONG SELL signals with confidence ≥ 60%
- [ ] Check risk calculations for any signals you want to act on

### During the trading session

- [ ] For interesting screener candidates, run the full analysis: `python main.py TICKER`
- [ ] Monitor open paper positions in the **Portfolio** page
- [ ] Note any CONFLICTING signals for manual review

### End of day

- [ ] Review the daily summary on the **Overview** page
- [ ] Check portfolio value change in the **Portfolio** page
- [ ] Update `config/watchlist.yaml` if needed (add/remove tickers)
- [ ] Archive or review log files in `scheduler/logs/`

---

## 2. Running the Scheduler

### Modes of operation

| Mode | Command | When to use |
|---|---|---|
| **Immediate run** | `python3 scheduler/daily_runner.py --now` | Manual trigger, testing, cron |
| **Daemon** | `python3 scheduler/daily_runner.py` | Leave running, fires at schedule.time |
| **Analysis only** | `python3 scheduler/daily_runner.py --now --no-execute` | Dry-run, no trade logging |

### Overriding the watchlist for a single run

```bash
# Analyse only specific tickers (ignores watchlist.yaml)
python3 scheduler/daily_runner.py --now --watchlist AAPL NVDA TSLA

# Use a different account balance
python3 scheduler/daily_runner.py --now --balance 50000

# With Telegram notifications
python3 scheduler/daily_runner.py --now --notify
```

### Checking if the daemon is running

```bash
ps aux | grep daily_runner.py
```

### Scheduler run status

Each run ends with one of three statuses:

| Status | Meaning |
|---|---|
| `success` | All tickers completed without errors |
| `partial` | Some tickers errored, others succeeded |
| `failed` | All tickers errored |

`partial` and `failed` runs trigger a failure email if email notifications are configured in `watchlist.yaml`.

### Checking the last scheduler run

Via the dashboard — go to the **Overview** page and look at the last run timestamp.

Via the database directly:

```bash
sqlite3 news_trading.db \
  "SELECT run_at, status, signals_generated, trades_executed, duration_seconds \
   FROM scheduler_logs ORDER BY id DESC LIMIT 5;"
```

---

## 3. Monitoring Performance

### Dashboard pages to check daily

| Page | What to look for |
|---|---|
| **Overview** | Portfolio total value trend, signals count, open positions |
| **Portfolio** | Unrealised P&L per position; large losses warrant manual review |
| **History** | Win/loss ratio on closed trades; adjust thresholds if consistently losing |
| **Backtesting** | Run monthly to check if the strategy is still working on recent data |

### Key metrics to track

| Metric | Healthy range | Action if outside |
|---|---|---|
| Signal distribution | 30–50% HOLD, balanced BUY/SELL | If always one direction, check newsAPI data quality |
| Confidence scores | Avg 40–70% | Very low avg → consider adjusting thresholds |
| Scheduler success rate | > 95% | Investigate recurring errors in logs |
| Average run duration | 30–120 seconds | Longer may indicate API timeouts |

### Portfolio health check

Run this query to see your current paper portfolio:

```bash
sqlite3 news_trading.db \
  "SELECT ticker, shares, avg_price, current_value FROM portfolio WHERE shares > 0;"
```

### Signal history for a specific ticker

```bash
sqlite3 news_trading.db \
  "SELECT combined_signal, confidence, created_at \
   FROM combined_signals WHERE ticker='AAPL' ORDER BY id DESC LIMIT 10;"
```

### Win rate calculation

```bash
sqlite3 news_trading.db \
  "SELECT
     COUNT(*) AS total_trades,
     SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
     ROUND(100.0 * SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_rate_pct,
     ROUND(AVG(pnl), 2) AS avg_pnl
   FROM trade_history
   WHERE action = 'SELL';"
```

---

## 4. Interpreting Signals

### What each combined signal means

**STRONG BUY** — Both news and technical indicators agree the stock is a buying opportunity.
- Sentiment: majority of headlines are bullish (avg score ≥ +0.30)
- Technical: at least one indicator is oversold or trending up (RSI < 30, MACD bullish crossover, or price below lower Bollinger Band)
- Action: system suggests opening a position sized by Half-Kelly

**WEAK BUY** — News is bullish but technical picture is neutral.
- Sentiment: bullish; Technical: HOLD
- Action: smaller position suggested (lower confidence)

**STRONG SELL** — Both news and technical indicators agree the stock is overvalued or under pressure.
- Sentiment: majority of headlines are bearish; Technical: overbought signals
- Action: system notes this for shorting (paper trading only tracks long positions)

**WEAK SELL** — News is bearish but technical picture is neutral.

**CONFLICTING** — The two signals disagree. This is the most common outcome.
- No position is suggested
- Worth monitoring — may resolve in a clear direction after more news

**HOLD** — No strong signal in either direction. The system takes no action.

### How to handle CONFLICTING signals manually

1. Read the individual headline scores in the **Agents** page
2. Check if the bearish/bullish headlines are credible and relevant
3. Look at the technical indicators — which is in stronger territory?
4. If you decide to act, you can manually run just one agent:
   ```bash
   python main.py AAPL --agent technical
   python main.py AAPL --agent sentiment
   ```

### Confidence thresholds

| Confidence | System action | Recommended manual action |
|---|---|---|
| ≥ 70% | Full-sized position | Consider following the signal |
| 50–69% | Reduced position | Review before acting |
| 30–49% | Minimal position | Extra caution; check the news manually |
| < 30% | No position | Do not act; wait for clarity |

---

## 5. Risk Management Guidelines

### System-enforced limits

The following limits are hard-coded in `agents/risk_agent.py` and cannot be overridden via configuration:

| Limit | Value | Purpose |
|---|---|---|
| Max per-trade allocation | 10% of account | Prevents over-concentration |
| Max risk per trade | 2% of account | Limits loss on any single trade |
| Minimum confidence | 30% | Prevents trading on noise |
| Stop-loss (STRONG signals) | −2% | Automatic loss limit |
| Stop-loss (WEAK signals) | −1% | Tighter limit for lower-conviction trades |
| Reward-to-risk ratio | 2:1 | Take-profit is always 2× the risk distance |

### Additional guidelines for real-money use

> **Note**: The system paper-trades only. These guidelines apply if you manually replicate signals with real money.

1. **Never risk more than 2% of your total capital on a single trade.** This matches the system's risk budget and is the industry standard for position sizing.

2. **Wait for STRONG signals with ≥ 60% confidence** before acting. WEAK signals have lower statistical significance.

3. **Do not override stop-losses.** The stop-loss levels are calculated to limit your maximum loss to 2% of account. Removing them eliminates this protection.

4. **Diversify across sectors.** Avoid having more than 3 positions in the same sector simultaneously.

5. **Reduce position size in high-volatility environments.** The system's historical data (`1mo` period) may not capture rapidly changing conditions. In high-VIX environments, consider halving the suggested position size.

6. **Do not trade CONFLICTING signals.** These represent genuine uncertainty in the data; forcing a trade in either direction has no edge.

7. **Review the screener results as a starting point, not a final decision.** High hotness score = unusual activity, not necessarily a good trade.

8. **Keep a trading journal.** Note why you acted on a signal, what happened, and what you would do differently. This is the most effective way to improve over time.

### Position sizing example

Account: $10,000 | Signal: STRONG BUY (75% confidence) | Stock price: $189.30

| Constraint | Calculation | Result |
|---|---|---|
| Kelly allocation | 10,000 × 8.5% | $850 |
| 10% cap | 10,000 × 10% | $1,000 |
| 2% risk budget | 10,000 × 2% / 2% stop | $10,000 × 2% / 0.02 = $10,000 |
| **Binding constraint** | Kelly ($850) | **$850** |
| Shares | 850 / 189.30 | **4 shares** |
| Stop-loss | 189.30 × (1 − 0.02) | **$185.52** |
| Take-profit | 189.30 × (1 + 0.04) | **$196.87** |
| Max risk | 4 × (189.30 − 185.52) | **$15.12** |

---

## 6. Database Maintenance

### Backup

The database is a single SQLite file. Back it up regularly:

```bash
cp news_trading.db "backups/news_trading_$(date +%Y%m%d).db"
```

Or set up an automated backup:

```bash
# Add to cron (runs daily at 23:00)
0 23 * * * cp /path/to/news_trading.db /path/to/backups/news_trading_$(date +%Y%m%d).db
```

### Size check

```bash
ls -lh news_trading.db
sqlite3 news_trading.db ".tables"
sqlite3 news_trading.db "SELECT name, COUNT(*) FROM sqlite_master JOIN pragma_table_info(name) GROUP BY name;"
```

### Pruning old data

If the database grows large, prune old screener results (safe to delete, easily regenerated):

```bash
sqlite3 news_trading.db \
  "DELETE FROM screener_results WHERE run_at < date('now', '-90 days');"
```

Prune old headline scores (safe; runs table keeps the summary):

```bash
sqlite3 news_trading.db \
  "DELETE FROM headline_scores \
   WHERE run_id IN (SELECT id FROM runs WHERE created_at < date('now', '-90 days'));"
```

### Schema migration

If you update the codebase and see "no such column" errors, run:

```bash
python3 -c "from storage.database import Database; Database()"
```

This triggers `_migrate_schema()` which safely adds any new columns.

---

## 7. Log Management

### Log file location

```
scheduler/logs/
├── 2026-02-18.log
├── 2026-02-17.log
└── ...
```

Each day gets one log file. The file contains DEBUG, INFO, WARNING, and ERROR messages from the entire scheduler run.

### Checking today's log

```bash
cat scheduler/logs/$(date +%Y-%m-%d).log
```

### Finding errors in logs

```bash
grep "ERROR\|WARNING" scheduler/logs/$(date +%Y-%m-%d).log
```

### Log rotation

Logs are not automatically rotated. To keep only the last 30 days:

```bash
# Run manually or add to cron
find scheduler/logs/ -name "*.log" -mtime +30 -delete
```

### Log levels

| Level | Meaning |
|---|---|
| `DEBUG` | Detailed per-ticker steps (only in log file, not terminal) |
| `INFO` | Normal progress messages (terminal + log file) |
| `WARNING` | Non-fatal issues (API timeout, ticker fetch failed) |
| `ERROR` | Fatal per-ticker errors (analysis could not complete) |

A `WARNING` for one ticker does not affect other tickers. Only `ERROR` entries indicate a ticker was skipped entirely.
