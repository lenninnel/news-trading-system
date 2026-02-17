# News Trading System

A multi-agent system that fetches financial headlines, scores their sentiment
using Claude, and produces a **BUY / SELL / HOLD** trading signal. Every
analysis run is persisted to a local SQLite database for auditing and
back-testing.

---

## Architecture

```
news-trading-system/
├── agents/
│   ├── base_agent.py        # Abstract base class for all agents
│   └── sentiment_agent.py   # Claude-powered headline classifier
├── data/
│   ├── news_feed.py         # NewsAPI headlines fetcher
│   └── market_data.py       # Yahoo Finance price/fundamentals fetcher
├── storage/
│   └── database.py          # SQLite persistence layer
├── orchestrator/
│   └── coordinator.py       # Wires everything into a single pipeline
├── config/
│   └── settings.py          # Environment variables & constants
├── main.py                  # CLI entry point
├── main_old.py              # Original single-file implementation (reference)
└── requirements.txt
```

### Pipeline

```
NewsFeed ──► SentimentAgent (×N headlines) ──► Aggregate ──► Signal
                                                                │
                                                           Database
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
NEWSAPI_KEY=your_newsapi_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
# Optional — defaults to news_trading.db in the working directory
# DB_PATH=news_trading.db
```

- **NEWSAPI_KEY** — get a free key at <https://newsapi.org>
- **ANTHROPIC_API_KEY** — get a key at <https://console.anthropic.com>

---

## Usage

```bash
python main.py <TICKER>
```

**Example:**

```bash
python main.py AAPL
```

**Sample output:**

```
  AAPL  |  Apple Inc.  |  189.30 USD

Fetching headlines for AAPL...
Found 10 headline(s).

[1/10] Analysing: Apple hits all-time high on strong iPhone demand
         [+] BULLISH — Strong consumer demand signals positive revenue growth.
[2/10] Analysing: Analysts warn of Apple supply chain risks
         [-] BEARISH — Supply disruptions could weigh on near-term earnings.
...

============================================================
  Ticker:       AAPL
  Run ID:       42
  Headlines:    10 analysed
  Bullish:      6
  Bearish:      2
  Neutral:      2
  Avg Score:    +0.40  (range -1.00 to +1.00)
  Signal:       BUY
============================================================
```

---

## Signal thresholds

| Avg Score      | Signal |
|----------------|--------|
| ≥ +0.30        | BUY    |
| ≤ −0.30        | SELL   |
| Between        | HOLD   |

Thresholds are configurable in `config/settings.py`.

---

## Database

Results are stored in `news_trading.db` (SQLite). Two tables:

- **runs** — one row per `main.py` invocation (ticker, signal, avg score, timestamp)
- **headline_scores** — one row per headline with sentiment, score, and Claude's reasoning

Query example:

```bash
sqlite3 news_trading.db "SELECT ticker, signal, avg_score, created_at FROM runs ORDER BY id DESC LIMIT 10;"
```

---

## Extending the system

- **Add a new data source**: create a class in `data/` (no base class required).
- **Add a new agent**: subclass `BaseAgent`, implement `name` and `run()`.
- **Change the model**: update `CLAUDE_MODEL` in `config/settings.py`.
- **Change thresholds**: update `BUY_THRESHOLD` / `SELL_THRESHOLD` in `config/settings.py`.
