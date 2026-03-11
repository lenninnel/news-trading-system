# Setup Guide — News Trading System

> **Audience**: Anyone installing the system for the first time on macOS.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Environment Setup](#3-environment-setup)
4. [First Run](#4-first-run)
5. [Daily Scheduler](#5-daily-scheduler)
6. [Dashboard](#6-dashboard)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

| Requirement | Minimum version | How to check |
|---|---|---|
| Python | 3.9+ | `python3 --version` |
| pip | 21+ | `pip3 --version` |
| Git | Any recent | `git --version` |
| macOS | 12+ | `sw_vers` |

You also need accounts at:
- [newsapi.org](https://newsapi.org) — free tier: 100 requests/day
- [console.anthropic.com](https://console.anthropic.com) — pay-per-token
- [alpaca.markets](https://alpaca.markets) — free paper-trading account (optional, for broker integration)

---

## 2. Installation

```bash
# Clone the repository
git clone https://github.com/lenninnel/news-trading-system.git
cd news-trading-system

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip3 install -r requirements.txt
```

---

## 3. Environment Setup

Copy the template and fill in your keys:

```bash
cp .env.template .env
```

Open `.env` in your editor and set at minimum:

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...your-key-here
NEWSAPI_KEY=your-newsapi-key-here

# Broker mode: paper_local | alpaca_paper | alpaca_live
TRADING_MODE=paper_local

# Alpaca API (required for alpaca_paper / alpaca_live modes)
ALPACA_API_KEY=your-alpaca-key
ALPACA_SECRET_KEY=your-alpaca-secret

# Optional — Reddit sentiment (via PRAW)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
```

**Key sources:**

| Key | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) → API Keys |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org) → Get API Key |
| `ALPACA_API_KEY` | [app.alpaca.markets](https://app.alpaca.markets) → Paper Trading → API Keys |
| `REDDIT_CLIENT_ID` | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) (optional) |

---

## 4. First Run

```bash
# Activate your virtual environment
source .venv/bin/activate

# Analyse a single stock (no trade execution)
python3 main.py AAPL

# Analyse and execute a paper trade
python3 main.py AAPL --execute

# Analyse with a custom account balance
python3 main.py NVDA --balance 25000 --execute
```

You should see:
- Market data (price, name)
- 10 headline sentiment scores
- Technical indicators (RSI, MACD, Bollinger Bands)
- A combined signal (STRONG BUY / WEAK BUY / HOLD / WEAK SELL / STRONG SELL)
- Position sizing from the risk agent

---

## 5. Daily Scheduler

The system includes an async batch runner that analyses multiple tickers in parallel.

### Activate the cron job

```bash
bash scheduler/install_cron.sh
```

This installs a cron entry that runs every weekday at 09:30 (local machine time):

```
30 9 * * 1-5 cd /Users/lennartwehrheim/news-trading-system && python3 -m scheduler.daily_runner --now >> logs/daily_runner.log 2>&1
```

### Verify it's running

```bash
# Check the cron entry exists
crontab -l

# Check the log file
cat logs/daily_runner.log

# Follow the log in real time
tail -f logs/daily_runner.log
```

### Run the screener manually

```bash
# Run immediately with default 20-stock watchlist
python3 -m scheduler.daily_runner --now

# Custom watchlist
python3 -m scheduler.daily_runner --now --watchlist AAPL,NVDA,TSLA,MSFT

# Adjust concurrency
python3 -m scheduler.daily_runner --now --workers 8
```

### Remove the cron job

```bash
crontab -e   # delete the line containing daily_runner
```

---

## 6. Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

The dashboard has four pages:
- **Overview** — latest signals, sentiment scores, and trade history
- **Technical** — RSI, MACD, Bollinger Band charts for each ticker
- **Risk** — position sizes, stop-loss/take-profit levels, Kelly fractions
- **Backtesting** — run walk-forward optimisation, view equity curves and Sharpe ratios

---

## 7. Troubleshooting

### "ModuleNotFoundError: No module named 'anthropic'"

Virtual environment is not active or dependencies not installed:

```bash
source .venv/bin/activate
pip3 install -r requirements.txt
```

### "NEWSAPI_KEY not set" or "ANTHROPIC_API_KEY not set"

The `.env` file is missing or not in the project root:

```bash
ls -la .env        # should exist
cat .env           # should contain your keys
```

### "yfinance: 404 No data found"

The ticker symbol is wrong. Use Yahoo Finance suffixes for non-US stocks:
- US: `AAPL`, `NVDA`
- Germany: `BAYN.DE`, `SAP.DE`
- UK: `SHEL.L`

### "Invalid Crumb" / 401 from yfinance

The system handles this automatically with a cache clear and retry. If it persists, delete the yfinance cache:

```bash
rm -rf ~/.cache/py-yfinance
```

### StockTwits 403 errors

The system auto-disables StockTwits after 3 consecutive failures and logs a warning. Analysis continues using NewsAPI and Reddit. No action needed.

### "streamlit: command not found"

```bash
source .venv/bin/activate
pip3 install streamlit
streamlit run dashboard/app.py
```

### Dashboard shows "no data"

Run at least one analysis first:

```bash
python3 main.py AAPL --execute
```

Then refresh the dashboard.
