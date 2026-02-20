# News Trading System

A multi-agent AI system that monitors financial news, analyses technical indicators, sizes positions with Kelly Criterion, and produces actionable **BUY / SELL / HOLD** signals — all running locally on your machine with a live Streamlit dashboard.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-202%20passed-brightgreen)](tests/)
[![Deploy on Railway](https://railway.app/button.svg)](docs/DEPLOYMENT.md)

---

## Features

| Category | What it does |
|---|---|
| **Sentiment Analysis** | Classifies up to 10 recent headlines per ticker using Claude (Anthropic) as bullish / bearish / neutral |
| **Technical Analysis** | Computes RSI-14, MACD, SMA-20/50, Bollinger Bands and derives a deterministic signal |
| **Signal Fusion** | Combines sentiment + technical into STRONG BUY / STRONG SELL / WEAK BUY / WEAK SELL / CONFLICTING / HOLD |
| **Risk Sizing** | Half-Kelly Criterion position sizing with stop-loss and 2:1 take-profit targets |
| **Paper Trading** | Simulated order execution and portfolio tracking stored in SQLite |
| **Stock Screener** | Scans 400+ tickers across DAX, MDAX, SDAX, TecDAX, S&P 500, NASDAQ 100, EURO STOXX 50, FTSE 100, CAC 40 and ranks by a composite hotness score |
| **Backtesting** | Day-by-day historical simulation with Sharpe ratio, max drawdown, win rate |
| **Daily Scheduler** | Runs the full pipeline automatically at market open (daemon or cron) |
| **Telegram Alerts** | Real-time trade signals and end-of-day summaries via Telegram bot |
| **Dashboard** | Interactive Streamlit UI with seven pages: Overview, Signals, Portfolio, History, Agents, Backtesting, Screener |

---

## Quick Start (5 Steps)

### Step 1 — Clone & enter the project

```bash
git clone https://github.com/your-username/news-trading-system.git
cd news-trading-system
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> Requires **Python 3.9+**. Using a virtual environment is recommended.

### Step 3 — Add your API keys

Create a `.env` file in the project root:

```env
NEWSAPI_KEY=your_newsapi_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

Get free keys at [newsapi.org](https://newsapi.org) and [console.anthropic.com](https://console.anthropic.com).

### Step 4 — Analyse your first ticker

```bash
python main.py AAPL
```

### Step 5 — Open the dashboard

```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

---

## Usage Examples

### Single-ticker analysis

```bash
# Full pipeline: sentiment + technical + risk sizing
python main.py AAPL

# Sentiment only
python main.py AAPL --agent sentiment

# Technical only
python main.py AAPL --agent technical

# With paper-trade execution and custom balance
python main.py AAPL --execute --balance 25000

# With Telegram notifications
python main.py AAPL --execute --notify
```

### Daily scheduler

```bash
# Run the full watchlist once immediately (great for cron jobs)
python3 scheduler/daily_runner.py --now

# Run as a daemon (fires at schedule.time in watchlist.yaml every weekday)
python3 scheduler/daily_runner.py

# Override watchlist and balance for a single run
python3 scheduler/daily_runner.py --now --watchlist AAPL NVDA TSLA --balance 50000

# Analysis only — no paper trades
python3 scheduler/daily_runner.py --now --no-execute
```

### Stock screener

```bash
# Scan German markets, return top 5
python3 -m agents.screener_agent --markets DE --top 5

# Full multi-market scan, focus on German stocks
python3 -m agents.screener_agent --markets US DE EU --focus DE --top 40

# US only, no DB logging, verbose output
python3 -m agents.screener_agent --markets US --top 20 --no-db -v
```

### Backtester

```bash
python3 backtest/engine.py --ticker AAPL \
    --start 2024-01-01 --end 2025-01-01 \
    --balance 10000 --sentiment random
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Entry Points                              │
│   main.py (CLI)   ·   daily_runner.py   ·   dashboard/app.py   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Coordinator (orchestrator/)                  │
│  Wires agents, data sources, execution layer, and notifications  │
└───────────┬──────────────┬──────────────┬───────────────────────┘
            │              │              │
            ▼              ▼              ▼
┌───────────────┐  ┌───────────────┐  ┌──────────────────────────┐
│  Data Layer   │  │  Agent Layer  │  │   Screener (standalone)  │
│               │  │               │  │                          │
│  NewsFeed     │  │  Sentiment    │  │  ScreenerAgent           │
│  (NewsAPI)    │  │  Agent        │  │  ┌ DAX / MDAX / SDAX     │
│               │  │  (Claude AI)  │  │  ├ TecDAX                │
│  MarketData   │  │               │  │  ├ S&P 500 / NASDAQ 100  │
│  (yfinance)   │  │  Technical    │  │  ├ EURO STOXX 50         │
│               │  │  Agent        │  │  ├ FTSE 100 / CAC 40     │
└───────────────┘  │  (yfinance+ta)│  │  └ Hotness score 0–10    │
                   │               │  └──────────────────────────┘
                   │  Risk Agent   │
                   │  (Kelly)      │
                   └───────┬───────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Storage Layer                                │
│              SQLite  ·  news_trading.db  (10 tables)            │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Execution & Notification Layer                       │
│         PaperTrader  ·  TelegramNotifier  ·  Email (SMTP)       │
└─────────────────────────────────────────────────────────────────┘
```

### Signal pipeline (per ticker)

```
NewsAPI headlines
      │
      ▼
SentimentAgent × N   ──►  avg_score  ──►  sentiment_signal (BUY/SELL/HOLD)
                                                     │
yfinance OHLCV                                       │
      │                                              ▼
TechnicalAgent  ──────────────────────►  combine_signals()
(RSI, MACD, BB,                               │
 SMA-20/50)                                   ▼
                               combined_signal (STRONG BUY … HOLD)
                                              │
                                         confidence score
                                              │
                                              ▼
                                        RiskAgent
                                   (half-Kelly position size,
                                    stop-loss, take-profit)
                                              │
                                              ▼
                                       PaperTrader
                                   (optional execution log)
                                              │
                                              ▼
                                     TelegramNotifier
                                     (optional alert)
```

---

## Configuration

Edit `config/watchlist.yaml` to customise the daily scheduler:

```yaml
watchlist:
  - AAPL
  - NVDA
  - TSLA
  - MSFT
  - GOOGL

account:
  balance: 10000.0         # Paper-trading account size (USD)

execution:
  enabled: true            # false = analysis only, no trade logs

schedule:
  time: "09:30"            # HH:MM local machine time
  weekdays_only: true

screener:
  markets: [US, DE, EU]
  focus_market: DE
  top_candidates: 40
```

All thresholds live in `config/settings.py`:

```python
BUY_THRESHOLD  = 0.30    # avg_score ≥ +0.30 → BUY
SELL_THRESHOLD = -0.30   # avg_score ≤ −0.30 → SELL
MAX_HEADLINES  = 10      # headlines per ticker from NewsAPI
```

---

## Signal Reference

### Sentiment score

| Avg Score | Signal |
|---|---|
| ≥ +0.30 | BUY |
| ≤ −0.30 | SELL |
| Between  | HOLD |

### Combined signal fusion

| Sentiment | Technical | Combined Signal |
|---|---|---|
| BUY | BUY | STRONG BUY |
| SELL | SELL | STRONG SELL |
| BUY | HOLD | WEAK BUY |
| SELL | HOLD | WEAK SELL |
| BUY | SELL | CONFLICTING |
| SELL | BUY | CONFLICTING |
| HOLD | any | HOLD |

### Position sizing rules

- **Minimum confidence**: 30% to open any position
- **Max position**: 10% of account balance
- **Max risk per trade**: 2% of account balance
- **Stop-loss**: 2% (STRONG signals) / 1% (WEAK signals)
- **Take-profit**: 2:1 reward-to-risk ratio

---

## Screenshots

### Terminal output

```
============================================================
  AAPL  |  Apple Inc.  |  $189.30
============================================================
  Sentiment signal:   BUY   (avg score: +0.45)
  Technical signal:   BUY   (RSI: 28.4 · MACD bullish crossover)
  Combined signal:    STRONG BUY
  Confidence:         78%
------------------------------------------------------------
  Position size:      $820.00  (4 shares)
  Stop-loss:          $185.52  (−2.0%)
  Take-profit:        $196.36  (+3.7%)
  Max risk:           $20.00
============================================================
```

### Dashboard pages

| Page | Contents |
|---|---|
| Overview | Portfolio KPIs, recent signals, open positions |
| Signals | Full combined signal history with filters |
| Portfolio | Open positions, P&L summary |
| History | All paper trades with date-range filter |
| Agents | Sentiment runs, technical signals, risk calculations |
| Backtesting | Run engine, equity curve, metrics |
| Screener | Live multi-market scan, hotness rankings |

---

## Project Structure

```
news-trading-system/
├── agents/                  # AI and analysis agents
│   ├── base_agent.py        # Abstract base class
│   ├── sentiment_agent.py   # Claude-powered headline classifier
│   ├── technical_agent.py   # RSI / MACD / BB / SMA signals
│   ├── risk_agent.py        # Half-Kelly position sizing
│   └── screener_agent.py    # Multi-market momentum screener
├── backtest/
│   └── engine.py            # Historical backtesting engine
├── config/
│   ├── settings.py          # Constants and environment variables
│   └── watchlist.yaml       # Scheduler and screener configuration
├── dashboard/
│   └── app.py               # Streamlit 7-page dashboard
├── data/
│   ├── news_feed.py         # NewsAPI headlines
│   └── market_data.py       # Yahoo Finance price data
├── execution/
│   └── paper_trader.py      # Simulated order execution
├── notifications/
│   └── telegram_bot.py      # Real-time Telegram alerts
├── orchestrator/
│   └── coordinator.py       # Pipeline wiring
├── scheduler/
│   ├── daily_runner.py      # Daemon / cron scheduler
│   └── install_cron.sh      # Cron setup helper
├── storage/
│   └── database.py          # SQLite persistence (10 tables)
├── tests/                   # Unit tests
├── docs/                    # Extended documentation
│   ├── USER_MANUAL.md
│   ├── ARCHITECTURE.md
│   ├── SETUP_GUIDE.md
│   ├── OPERATIONS.md
│   └── DEVELOPMENT.md
├── main.py                  # CLI entry point
├── requirements.txt
├── CHANGELOG.md
└── .env                     # API keys (git-ignored)
```

---

## Documentation

| Document | Audience | Contents |
|---|---|---|
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | End users | How to start, read signals, use the dashboard, FAQ |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | All | System design, agent descriptions, data flow, DB schema |
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | New users | Detailed install for Mac / Linux / Windows, API setup |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Operators | Daily checklist, monitoring, risk management |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Contributors | Code structure, adding agents, testing, git workflow |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Operators | Daily/weekly/monthly tasks, emergency procedures |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Operators | Common issues, diagnostic commands, escalation guide |
| [docs/COST_MONITORING.md](docs/COST_MONITORING.md) | Operators | API costs, Railway.app pricing, free-tier tips |
| [docs/DEPLOYMENT_CHECKLIST.md](docs/DEPLOYMENT_CHECKLIST.md) | DevOps | Pre/post-deployment verification checklist |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | DevOps | Railway.app and Docker deployment guide |

---

## Getting Started in Production

Moving from local development to a production deployment takes about 30 minutes. Here's the condensed path:

### 1. Verify everything works locally

```bash
# Run the full test suite — must be 202/202 green
python3 -m pytest tests/ -q

# Smoke test: single ticker, no execution
python3 main.py AAPL --no-execute
```

### 2. Work through the deployment checklist

Open [docs/DEPLOYMENT_CHECKLIST.md](docs/DEPLOYMENT_CHECKLIST.md) and check off every item before pushing. Key gates:

```bash
# Environment variables
python3 -c "import os; print('Keys set:', bool(os.environ.get('ANTHROPIC_API_KEY')), bool(os.environ.get('NEWSAPI_KEY')))"

# Database healthy
python3 -c "from storage.database import Database; Database(); print('DB OK')"

# Kill switch works
python3 main.py --kill-switch status
```

### 3. Deploy to Railway.app (recommended)

```bash
railway login
railway link
railway variables set ANTHROPIC_API_KEY=<key> NEWSAPI_KEY=<key>
git push origin main   # triggers auto-deploy
```

Full deployment steps: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

### 4. Set up monitoring and alerting

```bash
# Telegram alerts (optional but recommended)
railway variables set TELEGRAM_BOT_TOKEN=<token> TELEGRAM_CHAT_ID=<id>

# Verify a test message arrives
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
  -d "chat_id=$TELEGRAM_CHAT_ID&text=Production+deployed+✓"
```

### 5. Ongoing operations

| Reference | Use for |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Daily/weekly tasks, emergency procedures |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Diagnosing and fixing common issues |
| [docs/COST_MONITORING.md](docs/COST_MONITORING.md) | Tracking API and infrastructure spend |

### API fallback resilience

The system includes a 4-level fallback chain for both news and price data — if the primary source fails, it automatically drops to the next level and logs the degradation:

```
News:  NewsAPI → Yahoo/Nasdaq RSS → Google News RSS → 24h cache
Price: yfinance → Alpha Vantage   → Yahoo JSON API  → cached price
```

Check fallback status at any time:
```bash
python3 -c "
from data.fallback_coordinator import FallbackCoordinator
import json
print(json.dumps(FallbackCoordinator.get_status(), indent=2))
"
```

---

## Contributing

Contributions are welcome. Please read [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) before opening a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for new logic in `tests/`
4. Open a pull request against `main`

Please use the [issue template](.github/ISSUE_TEMPLATE.md) for bug reports and feature requests.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 News Trading System Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

> **Disclaimer**: This software is for educational and research purposes only. It does not constitute financial advice. Always do your own research before making investment decisions.
