# Architecture — News Trading System

> **Audience**: Developers, researchers, and technically curious users.

---

## Table of Contents

1. [System Design Overview](#1-system-design-overview)
2. [Agent Descriptions](#2-agent-descriptions)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [Database Schema](#4-database-schema)
5. [Technology Stack](#5-technology-stack)
6. [Key Design Decisions](#6-key-design-decisions)

---

## 1. System Design Overview

The News Trading System is a **modular, agent-based pipeline** that processes market data in four layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — Entry Points                                          │
│  main.py   ·   daily_runner.py   ·   dashboard/app.py           │
│  screener_agent (CLI)   ·   backtest/engine.py                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2 — Orchestration                                         │
│  orchestrator/coordinator.py                                     │
│  Wires agents, data, execution, and notifications together       │
└──────────┬────────────────────────────────────────┬─────────────┘
           │                                        │
           ▼                                        ▼
┌───────────────────────┐              ┌────────────────────────┐
│  Layer 3 — Agents     │              │  Layer 3 — Data        │
│                       │              │                        │
│  SentimentAgent       │              │  NewsFeed              │
│  TechnicalAgent       │              │  (NewsAPI REST)        │
│  RiskAgent            │              │                        │
│  ScreenerAgent        │              │  MarketData            │
│                       │              │  (yfinance)            │
└──────────┬────────────┘              └────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4 — Infrastructure                                        │
│  storage/database.py   ·   execution/paper_trader.py            │
│  notifications/telegram_bot.py   ·   scheduler/daily_runner.py  │
└─────────────────────────────────────────────────────────────────┘
```

### Core architectural principles

| Principle | Implementation |
|---|---|
| **Single responsibility** | Each agent does exactly one thing; coordination is in `Coordinator` |
| **Pluggable agents** | All agents inherit from `BaseAgent`; swapping one does not break others |
| **Local-first storage** | SQLite — no external database required |
| **Fail-safe pipeline** | Exceptions in one ticker do not abort the scheduler run for others |
| **Idempotent schema** | `CREATE TABLE IF NOT EXISTS` + `_migrate_schema()` handle upgrades |

---

## 2. Agent Descriptions

### BaseAgent (agents/base_agent.py)

Abstract base class that every agent must implement.

```python
class BaseAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...          # Human-readable identifier

    @abstractmethod
    def run(self, *args, **kwargs): ... # Primary execution method
```

---

### SentimentAgent (agents/sentiment_agent.py)

**Role**: Classifies a single news headline as bullish, bearish, or neutral using Claude AI.

**Input**: `(headline: str, ticker: str)`

**Output**:
```json
{
  "sentiment": "bullish",
  "score": 1,
  "reason": "Strong revenue growth expectations signal positive market momentum.",
  "headline": "Apple hits all-time high on strong iPhone demand"
}
```

**How it works**:
1. Builds a structured prompt with the ticker and headline
2. Sends to Claude (`claude-sonnet-4-6`) via the Anthropic API
3. Parses JSON response: `sentiment` (bullish/bearish/neutral), `score` (+1/0/−1), `reason`
4. Falls back to HOLD on parse errors

**Signal thresholds** (applied by Coordinator):
- Average score ≥ +0.30 → BUY
- Average score ≤ −0.30 → SELL
- Between → HOLD

---

### TechnicalAgent (agents/technical_agent.py)

**Role**: Derives a BUY / SELL / HOLD signal from price-based technical indicators.

**Input**: `(ticker: str)`

**Output**:
```json
{
  "ticker": "AAPL",
  "signal": "BUY",
  "reasoning": "RSI oversold (28.4); MACD bullish crossover",
  "indicators": {
    "rsi": 28.4,
    "macd": 0.32,
    "macd_signal": -0.15,
    "sma_20": 187.5,
    "sma_50": 183.2,
    "bb_upper": 195.0,
    "bb_lower": 180.0,
    "price": 189.3
  },
  "signal_id": 42
}
```

**Indicators computed**:

| Indicator | Period | Library |
|---|---|---|
| RSI | 14 days | `ta` |
| MACD | 12/26/9 | `ta` |
| SMA | 20 and 50 days | `ta` |
| Bollinger Bands | 20 days, 2σ | `ta` |

**Signal rules** (deterministic, in priority order):

| Condition | Signal |
|---|---|
| RSI < 30 | BUY |
| RSI > 70 | SELL |
| MACD line crosses above MACD signal line | BUY |
| MACD line crosses below MACD signal line | SELL |
| Price < lower Bollinger Band | BUY |
| Price > upper Bollinger Band | SELL |
| None of above | HOLD |

---

### RiskAgent (agents/risk_agent.py)

**Role**: Determines how much money to deploy on a signal using the Half-Kelly Criterion.

**Input**: `(ticker, signal, confidence, current_price, account_balance)`

**Output**:
```json
{
  "position_size_usd": 820.0,
  "shares": 4,
  "stop_loss": 185.52,
  "take_profit": 196.36,
  "risk_amount": 20.0,
  "kelly_fraction": 0.082,
  "stop_pct": 2.0,
  "skipped": false,
  "skip_reason": null,
  "calc_id": 17
}
```

**Kelly Criterion implementation**:

```
p = 0.50 + (confidence / 100 × 0.30)   # Win probability estimate
q = 1 - p                               # Loss probability
b = 2.0                                 # Win/loss ratio (2:1 R:R)

kelly_full = (p × b − q) / b           # Full Kelly
kelly_half = kelly_full / 2             # Half-Kelly (conservative)
```

**Position size** = minimum of:
1. `account_balance × kelly_half`
2. `account_balance × 0.10` (10% cap per trade)
3. `account_balance × 0.02 / stop_pct` (2% max risk budget)

**Stop-loss and take-profit**:

| Signal strength | Stop-loss | Take-profit |
|---|---|---|
| STRONG BUY / STRONG SELL | −2% from entry | +4% from entry (2:1 ratio) |
| WEAK BUY / WEAK SELL | −1% from entry | +2% from entry (2:1 ratio) |

**Safety gates** (trade is skipped if any apply):
- Signal is HOLD or CONFLICTING
- Confidence < 30%
- Calculated shares rounds to 0

---

### ScreenerAgent (agents/screener_agent.py)

**Role**: Scans 400+ stocks across multiple indices and ranks by a composite "hotness" score.

**Input**: `(markets: list[str], focus_market: str, top: int)`

**Output**:
```json
{
  "run_at": "2026-02-18T09:35:00Z",
  "markets_scanned": ["US", "DE"],
  "focus_market": "DE",
  "universe_size": 229,
  "screened": 12,
  "candidates": [
    {
      "ticker": "BAYN.DE",
      "market": "DAX",
      "country": "DE",
      "hotness": 6.65,
      "price_change": -7.11,
      "volume_ratio": 4.2,
      "rsi": 51.4
    }
  ]
}
```

**Universes**:

| Market group | Indices | Approximate tickers |
|---|---|---|
| DE | DAX 40, MDAX (~50), SDAX (~70), TecDAX (30) | 130 |
| US | S&P 500, NASDAQ 100 | 600 |
| EU | EURO STOXX 50, FTSE 100, CAC 40 | 120 |

**Hotness formula** (all components normalised to [0, 1]):

```
hotness = (vol_ratio_norm  × 0.30)
        + (price_chg_norm  × 0.30)
        + (rsi_extreme     × 0.20)
        + (liquidity_norm  × 0.10)
        + (market_priority × 0.10)
× 10
```

| Component | Normalisation |
|---|---|
| Volume ratio | `min(vol_ratio / 5.0, 1.0)` — cap at 5× |
| Price change | `min(abs(price_change) / 10.0, 1.0)` — cap at 10% |
| RSI extreme | `abs(rsi − 50) / 50` — distance from neutral |
| Liquidity | `min(log10(avg_volume) / log10(10M), 1.0)` |
| Market priority | DE=+0.2, US=0.0, EU/non-DE=−0.1 → mapped to [0, 1] |

**Filter thresholds**:

| Market type | Min volume ratio | Min price change | Min avg volume |
|---|---|---|---|
| blue_chip | 2.0× | 3.0% | — |
| mid_cap | 1.5× | 4.0% | 100,000 |
| small_cap | 1.5× | 4.0% | 100,000 |

**Focus market quota**: When `focus_market="DE"`, at least `max(10, top÷3)` slots are reserved for German stocks.

---

### Coordinator (orchestrator/coordinator.py)

**Role**: Central pipeline that wires all agents, data sources, and infrastructure together.

**Single-agent pipeline** (`run()`):
```
NewsFeed.fetch(ticker)
  → [headline₁, headline₂, …, headlineN]
  → SentimentAgent.run(headline, ticker) × N
  → aggregate scores
  → derive signal (BUY/SELL/HOLD)
  → Database.log_run()
```

**Combined pipeline** (`run_combined()`):
```
Coordinator.run(ticker)          # sentiment pipeline
  → TechnicalAgent.run(ticker)
  → combine_signals()            # fusion matrix
  → confidence()                 # 0–1 score
  → RiskAgent.run()              # position sizing
  → PaperTrader.track_trade()    # optional execution
  → TelegramNotifier.send()      # optional alert
  → Database.log_combined_signal()
```

**Signal fusion matrix**:

```python
def combine_signals(sentiment_signal, technical_signal):
    if sentiment_signal == "BUY"  and technical_signal == "BUY":  return "STRONG BUY"
    if sentiment_signal == "SELL" and technical_signal == "SELL": return "STRONG SELL"
    if sentiment_signal == "BUY"  and technical_signal == "HOLD": return "WEAK BUY"
    if sentiment_signal == "SELL" and technical_signal == "HOLD": return "WEAK SELL"
    if sentiment_signal == technical_signal == "HOLD":            return "HOLD"
    return "CONFLICTING"
```

**Confidence calculation**:

| Signal | Formula | Range |
|---|---|---|
| STRONG BUY / STRONG SELL | `0.6 + (abs(avg_score) × 0.4)` | [0.60, 1.00] |
| WEAK BUY / WEAK SELL | `0.2 + (abs(avg_score) × 0.4)` | [0.20, 0.60] |
| CONFLICTING | Fixed 0.10 | 0.10 |
| HOLD | Fixed 0.25 | 0.25 |

---

## 3. Data Flow Diagram

```
External Sources           Application Core              Storage
─────────────────          ────────────────              ───────

NewsAPI ──────────────►  NewsFeed.fetch()
                              │
                              ▼
                    SentimentAgent.run() ─────────────► runs
                         (× headlines)                  headline_scores
                              │
                              ▼ sentiment_signal
                         Coordinator
                         .combine_signals() ──────────► combined_signals
                              │
                              ▼ combined_signal
Yahoo Finance ────────►  TechnicalAgent.run() ────────► technical_signals
  (OHLCV 3mo)                 │
                              ▼ technical_signal
                          RiskAgent.run() ──────────────► risk_calculations
                              │
                              ▼ position_size / shares
                          PaperTrader ─────────────────► portfolio
                         .track_trade()                  trade_history
                              │
                              ▼
                      TelegramNotifier
                         .send_signal()
                         (if configured)
```

### Screener data flow

```
Yahoo Finance ─────────► ScreenerAgent ──► screener_results
  (OHLCV 1mo,                 │
   batch downloads)           ▼
                     [ranked candidates]
                           │
                     Dashboard/Screener page
                     CLI output
```

---

## 4. Database Schema

All data is stored in `news_trading.db` (SQLite 3). The schema is initialised by `Database._init_schema()` and migrated forward by `Database._migrate_schema()`.

### Table: `runs`

Sentiment analysis execution log.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| ticker | TEXT | Stock symbol |
| headlines_fetched | INTEGER | Total headlines from NewsAPI |
| headlines_analysed | INTEGER | Successfully scored headlines |
| avg_score | REAL | Average sentiment score (−1 to +1) |
| signal | TEXT | BUY / SELL / HOLD |
| created_at | TEXT | ISO-8601 UTC |

### Table: `headline_scores`

Individual headline sentiment scores, linked to `runs`.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| run_id | INTEGER FK | → runs.id |
| headline | TEXT | Original headline |
| sentiment | TEXT | bullish / bearish / neutral |
| score | INTEGER | +1 / 0 / −1 |
| reason | TEXT | Claude's one-sentence explanation |

### Table: `technical_signals`

Technical analysis results.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| ticker | TEXT | |
| signal | TEXT | BUY / SELL / HOLD |
| rsi | REAL | RSI-14 |
| macd | REAL | MACD line |
| macd_signal | REAL | MACD signal line |
| macd_hist | REAL | MACD histogram |
| sma_20 | REAL | 20-period SMA |
| sma_50 | REAL | 50-period SMA |
| bb_upper | REAL | Upper Bollinger Band |
| bb_lower | REAL | Lower Bollinger Band |
| price | REAL | Latest close price |
| reasoning | TEXT | Human-readable trigger list |
| created_at | TEXT | ISO-8601 UTC |

### Table: `combined_signals`

Fused sentiment + technical signals.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| ticker | TEXT | |
| combined_signal | TEXT | STRONG BUY / STRONG SELL / WEAK BUY / WEAK SELL / CONFLICTING / HOLD |
| sentiment_signal | TEXT | BUY / SELL / HOLD |
| technical_signal | TEXT | BUY / SELL / HOLD |
| sentiment_score | REAL | avg_score from runs |
| confidence | REAL | 0.0–1.0 |
| run_id | INTEGER FK | → runs.id |
| technical_id | INTEGER FK | → technical_signals.id |
| created_at | TEXT | ISO-8601 UTC |

### Table: `risk_calculations`

Position sizing and risk parameter log.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| ticker | TEXT | |
| signal | TEXT | Combined signal string |
| confidence | REAL | 0–100 |
| current_price | REAL | Price at calculation time |
| account_balance | REAL | Account size used |
| position_size_usd | REAL | Dollar value to deploy |
| shares | INTEGER | Whole shares |
| stop_loss | REAL | Stop-loss price (NULL if skipped) |
| take_profit | REAL | Take-profit price (NULL if skipped) |
| risk_amount | REAL | Max dollar loss |
| kelly_fraction | REAL | Half-Kelly fraction |
| stop_pct | REAL | Stop-loss % (NULL if skipped) |
| skipped | INTEGER | 1 = no position; 0 = position |
| skip_reason | TEXT | Why skipped (NULL otherwise) |
| created_at | TEXT | ISO-8601 UTC |

### Table: `portfolio`

Current open positions (managed by PaperTrader).

| Column | Type | Notes |
|---|---|---|
| ticker | TEXT PK | Stock symbol |
| shares | INTEGER | Total shares held |
| avg_price | REAL | Average entry price |
| current_value | REAL | Shares × last known price |

### Table: `trade_history`

Complete paper-trade log.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| timestamp | TEXT | ISO-8601 UTC |
| ticker | TEXT | |
| action | TEXT | BUY / SELL |
| shares | INTEGER | |
| price | REAL | Execution price |
| stop_loss | REAL | |
| take_profit | REAL | |
| pnl | REAL | Realised P&L (SELL trades only) |

### Table: `scheduler_logs`

Daily scheduler run summaries.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| run_at | TEXT | ISO-8601 UTC start time |
| tickers | TEXT | JSON array of analysed tickers |
| signals_generated | INTEGER | |
| trades_executed | INTEGER | |
| portfolio_value | REAL | Total portfolio at end of run |
| duration_seconds | REAL | Wall-clock run time |
| errors | TEXT | JSON array of error strings |
| status | TEXT | success / partial / failed |
| summary | TEXT | Human-readable one-pager |
| created_at | TEXT | ISO-8601 UTC |

### Table: `backtest_results`

Historical backtest run summaries.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| ticker | TEXT | |
| start_date | TEXT | YYYY-MM-DD |
| end_date | TEXT | YYYY-MM-DD |
| initial_balance | REAL | Starting capital |
| final_balance | REAL | Ending capital |
| total_return_pct | REAL | Strategy return % |
| buy_and_hold_return_pct | REAL | Passive return % over same period |
| sharpe_ratio | REAL | Annualised Sharpe |
| max_drawdown_pct | REAL | Maximum drawdown % |
| win_rate_pct | REAL | % of winning closed trades |
| avg_win | REAL | Average profit per winning trade |
| avg_loss | REAL | Average loss per losing trade |
| total_trades | INTEGER | Closed trade count |
| sentiment_mode | TEXT | random / bullish / bearish / neutral |
| trades_json | TEXT | JSON array of individual trades |
| created_at | TEXT | ISO-8601 UTC |

### Table: `screener_results`

Stock screener candidate log. Unique index on `(run_at, ticker)`.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| run_at | TEXT | ISO-8601 UTC run timestamp |
| ticker | TEXT | |
| name | TEXT | Company name |
| market | TEXT | DAX / MDAX / SP500 / etc. |
| exchange | TEXT | XETRA / NYSE / LSE / etc. |
| country | TEXT | DE / US / GB / FR / EU |
| hotness | REAL | Composite score 0–10 |
| price | REAL | Latest close price |
| price_change | REAL | Today vs. yesterday (%) |
| volume_ratio | REAL | Today vs. 20-day average |
| volume | REAL | Today's volume |
| rsi | REAL | RSI-14 |
| market_cap | REAL | Market capitalisation (USD) |
| avg_volume | REAL | 20-day average volume |
| metrics | TEXT | Full metrics dict as JSON |
| created_at | TEXT | ISO-8601 UTC |

---

## 5. Technology Stack

### Runtime

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.9+ |
| AI model | Claude (Anthropic) | claude-sonnet-4-6 |
| Database | SQLite | 3 (stdlib) |
| Dashboard | Streamlit | Latest |
| Charts | Plotly | Latest |

### Data libraries

| Library | Purpose |
|---|---|
| `yfinance` | OHLCV price data from Yahoo Finance |
| `ta` | Technical indicators (RSI, MACD, BB, SMA) |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |

### External APIs

| API | Usage | Auth |
|---|---|---|
| [Anthropic](https://console.anthropic.com) | Claude sentiment analysis | `ANTHROPIC_API_KEY` |
| [NewsAPI](https://newsapi.org) | Financial headlines | `NEWSAPI_KEY` |
| [Yahoo Finance](https://finance.yahoo.com) | OHLCV data, fundamentals | None (public) |
| [Telegram Bot API](https://core.telegram.org/bots/api) | Real-time alerts | `TELEGRAM_BOT_TOKEN` |

### Infrastructure

| Component | Technology |
|---|---|
| Configuration | YAML (`config/watchlist.yaml`) |
| Secrets | `.env` + `python-dotenv` |
| Scheduling | `schedule` package or system cron |
| Notifications | Telegram HTTP API, SMTP (email) |
| Logging | Python `logging` module → daily log files |

---

## 6. Key Design Decisions

### Why SQLite?

SQLite requires zero setup, stores everything in a single file, and handles the write volumes of a daily trading system comfortably. The 5-second connection timeout and `check_same_thread=False` allow concurrent access from the Streamlit dashboard and CLI without deadlocks.

### Why Half-Kelly instead of Full Kelly?

Full Kelly is theoretically optimal but highly volatile in practice. Half-Kelly cuts the optimal position size by 50%, which dramatically reduces drawdowns with only a modest reduction in long-run growth rate. It is the standard in algorithmic trading for handling imprecise probability estimates.

### Why paper trading only?

Connecting to a live broker introduces significant regulatory, liability, and security concerns. Paper trading lets users validate and tune the strategy with zero financial risk before deciding whether to manually replicate trades with real money.

### Why is SentimentAgent stateless?

Each headline is classified independently so the agent can be called in isolation for testing, and to prevent one headline's context from biasing another's score. The Coordinator is responsible for aggregation.

### Why batch yfinance downloads?

yfinance has unofficial rate limits. Downloading tickers in batches of 50 with a brief pause between chunks reduces the risk of being temporarily blocked. The 5-minute in-memory cache prevents redundant downloads when the same ticker appears in multiple indices.
