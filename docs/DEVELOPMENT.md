# Development Guide — News Trading System

> **Audience**: Developers who want to contribute, extend the system, or understand the codebase.

---

## Table of Contents

1. [Code Structure](#1-code-structure)
2. [Development Environment Setup](#2-development-environment-setup)
3. [How to Add a New Agent](#3-how-to-add-a-new-agent)
4. [How to Add a New Data Source](#4-how-to-add-a-new-data-source)
5. [How to Add a New Database Table](#5-how-to-add-a-new-database-table)
6. [How to Add a New Dashboard Page](#6-how-to-add-a-new-dashboard-page)
7. [Testing Guidelines](#7-testing-guidelines)
8. [Git Workflow](#8-git-workflow)
9. [Code Conventions](#9-code-conventions)

---

## 1. Code Structure

```
news-trading-system/
│
├── agents/                      # AI and analysis agents
│   ├── base_agent.py            # Abstract base class — start here
│   ├── sentiment_agent.py       # Claude-powered sentiment classifier
│   ├── technical_agent.py       # RSI / MACD / BB / SMA signal engine
│   ├── risk_agent.py            # Half-Kelly position sizing
│   └── screener_agent.py        # Multi-market momentum screener
│
├── backtest/
│   └── engine.py                # Day-by-day backtesting with Plotly charts
│
├── config/
│   ├── settings.py              # Constants and .env loading
│   └── watchlist.yaml           # Scheduler and screener YAML config
│
├── dashboard/
│   └── app.py                   # Streamlit 7-page dashboard (~1300 lines)
│
├── data/
│   ├── news_feed.py             # NewsAPI adapter
│   └── market_data.py           # yfinance adapter (price + fundamentals)
│
├── execution/
│   └── paper_trader.py          # Simulated order execution & portfolio tracking
│
├── notifications/
│   ├── telegram_bot.py          # Telegram Bot API adapter
│   └── README.md                # Telegram setup walkthrough
│
├── orchestrator/
│   └── coordinator.py           # Pipeline wiring — the "main brain"
│
├── scheduler/
│   ├── daily_runner.py          # Daemon / cron scheduler
│   └── install_cron.sh          # Cron installation helper
│
├── storage/
│   └── database.py              # SQLite persistence (10 tables)
│
├── tests/
│   ├── test_coordinator.py
│   ├── test_risk_agent.py
│   └── test_technical_agent.py
│
├── docs/                        # Documentation
├── main.py                      # CLI entry point
└── requirements.txt
```

### Dependency graph

```
main.py
  └── Coordinator
        ├── NewsFeed (→ NewsAPI)
        ├── MarketData (→ yfinance)
        ├── SentimentAgent (→ Anthropic API)
        ├── TechnicalAgent (→ yfinance + ta)
        ├── RiskAgent
        ├── PaperTrader (→ Database)
        ├── TelegramNotifier (→ Telegram API)
        └── Database (→ SQLite)

daily_runner.py
  └── Coordinator (same as above)
  └── Database

screener_agent.py (standalone)
  └── yfinance
  └── ta
  └── Database
```

---

## 2. Development Environment Setup

```bash
# 1. Clone
git clone https://github.com/your-username/news-trading-system.git
cd news-trading-system

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Copy .env and add your keys
cp .env.example .env   # (or create manually)
nano .env

# 5. Verify
python main.py AAPL
python3 -m pytest tests/ -v
```

---

## 3. How to Add a New Agent

All agents must inherit from `BaseAgent` and implement two methods.

### Step 1 — Create the agent file

```python
# agents/my_agent.py
from __future__ import annotations
from typing import Any
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "MyAgent"

    def run(self, ticker: str, **kwargs: Any) -> dict:
        """
        Analyse the ticker and return a result dict.

        Args:
            ticker: Stock symbol.

        Returns:
            dict with at minimum: {signal, reasoning}
        """
        # ... your analysis logic here ...
        return {
            "ticker":    ticker,
            "signal":    "BUY",      # BUY / SELL / HOLD
            "reasoning": "...",
        }
```

### Step 2 — Integrate into the Coordinator (optional)

If your agent should be part of the main pipeline, add it to `orchestrator/coordinator.py`:

```python
# In Coordinator.__init__ or run_combined()
from agents.my_agent import MyAgent

my_agent = MyAgent()
my_result = my_agent.run(ticker)
```

### Step 3 — Add a database table (optional)

If you want to persist results, add a table to `storage/database.py`. See [How to Add a New Database Table](#5-how-to-add-a-new-database-table).

### Step 4 — Expose via CLI (optional)

Add a `main()` function and `if __name__ == "__main__":` guard for standalone CLI access. See `agents/screener_agent.py` for a complete example.

### Step 5 — Write tests

```python
# tests/test_my_agent.py
from agents.my_agent import MyAgent

def test_my_agent_returns_signal():
    agent = MyAgent()
    result = agent.run("AAPL")
    assert result["signal"] in ("BUY", "SELL", "HOLD")
    assert "ticker" in result
    assert "reasoning" in result
```

---

## 4. How to Add a New Data Source

Data sources are simple classes — no base class required.

```python
# data/my_source.py
from __future__ import annotations
import requests
from config.settings import MY_API_KEY   # add to settings.py

class MyDataSource:

    BASE_URL = "https://api.example.com/v1"

    def fetch(self, ticker: str) -> list[str]:
        """
        Fetch data for ticker.

        Returns:
            List of strings (e.g. headlines, data points).

        Raises:
            requests.HTTPError: On 4xx / 5xx responses.
            requests.Timeout:   On network timeout.
        """
        resp = requests.get(
            f"{self.BASE_URL}/data",
            params={"q": ticker, "apiKey": MY_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        return [item["text"] for item in resp.json()["articles"]]
```

Add your API key to `config/settings.py`:

```python
MY_API_KEY = os.getenv("MY_API_KEY", "")
```

And to `.env`:

```env
MY_API_KEY=your_key_here
```

---

## 5. How to Add a New Database Table

All schema changes go in `storage/database.py`.

### Step 1 — Add to `_init_schema()`

```python
# In Database._init_schema(), inside the executescript string:
"""
CREATE TABLE IF NOT EXISTS my_table (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker     TEXT    NOT NULL,
    my_column  REAL,
    created_at TEXT    NOT NULL
);
"""
```

### Step 2 — Add migration for existing databases

```python
# In Database._migrate_schema():
for col_def in ("my_column REAL",):
    try:
        conn.execute(f"ALTER TABLE my_table ADD COLUMN {col_def}")
    except sqlite3.OperationalError:
        pass  # already exists
```

### Step 3 — Add a log method

```python
def log_my_result(self, ticker: str, my_value: float) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with self._connect() as conn:
        cur = conn.execute(
            "INSERT INTO my_table (ticker, my_column, created_at) VALUES (?, ?, ?)",
            (ticker, my_value, now),
        )
        return cur.lastrowid
```

### Step 4 — Add a read method (optional, for dashboard)

```python
def get_my_results(self, limit: int = 100) -> list[dict]:
    with self._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM my_table ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
```

---

## 6. How to Add a New Dashboard Page

The dashboard is one file: `dashboard/app.py`. Pages are controlled by a sidebar radio button.

### Step 1 — Add to the navigation list

Find the `st.radio` call and add your page name:

```python
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Signals", "Portfolio", "History",
     "Agents", "Backtesting", "Screener", "My New Page"],  # add here
)
```

### Step 2 — Add the page block

At the end of `app.py`, add an `elif` block:

```python
elif page == "My New Page":
    st.title("My New Page")

    # Load data
    db = Database()
    data = db.get_my_results()

    if not data:
        st.info("No data yet. Run the analysis first.")
    else:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

        # Optional chart
        fig = px.bar(df, x="ticker", y="my_column", title="My Chart")
        st.plotly_chart(fig, use_container_width=True)
```

### Dashboard conventions

| Convention | Details |
|---|---|
| Data loading | Use `@st.cache_data(ttl=60)` for database reads |
| Error handling | Always check for empty results before rendering charts |
| Layout | Use `st.columns()` for KPI strips; `st.tabs()` for sub-sections |
| Downloads | Always offer a CSV download via `st.download_button()` |
| Charts | Use Plotly (`px` or `go`) for consistency |

---

## 7. Testing Guidelines

### Running tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run a specific test file
python3 -m pytest tests/test_risk_agent.py -v

# Run with coverage
python3 -m pytest tests/ --cov=agents --cov-report=term-missing
```

### Test structure

```
tests/
├── test_coordinator.py      # Integration-style tests for the full pipeline
├── test_risk_agent.py       # Unit tests for position sizing logic
└── test_technical_agent.py  # Unit tests for indicator computations
```

### Writing tests

**Test naming**: `test_<what>_<when>_<expected_result>`

```python
def test_risk_agent_skips_when_confidence_below_30():
    agent = RiskAgent()
    result = agent.run(
        ticker="AAPL",
        signal="STRONG BUY",
        confidence=25.0,        # below threshold
        current_price=100.0,
        account_balance=10000.0,
    )
    assert result["skipped"] is True
    assert result["position_size_usd"] == 0.0
```

**Mocking external APIs**: Use `unittest.mock.patch` to prevent real API calls in tests:

```python
from unittest.mock import patch, MagicMock

@patch("agents.sentiment_agent.anthropic.Anthropic")
def test_sentiment_agent_parses_bullish(mock_anthropic):
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_client.messages.create.return_value.content[0].text = (
        '{"sentiment": "bullish", "score": 1, "reason": "Strong results"}'
    )

    agent = SentimentAgent()
    result = agent.run("Apple hits record high", "AAPL")
    assert result["sentiment"] == "bullish"
    assert result["score"] == 1
```

### What to test

| Component | Test type | Priority |
|---|---|---|
| `RiskAgent` — position sizing math | Unit | High |
| `TechnicalAgent` — signal rules | Unit | High |
| `Coordinator.combine_signals` | Unit | High |
| `Coordinator.confidence` | Unit | High |
| `Database` — schema and CRUD | Integration | Medium |
| `ScreenerAgent.compute_hotness` | Unit | Medium |
| `SentimentAgent` (mocked API) | Unit | Medium |

---

## 8. Git Workflow

### Branching strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, deployable code — always passing tests |
| `feature/<name>` | New features |
| `fix/<name>` | Bug fixes |
| `docs/<name>` | Documentation-only changes |

### Workflow

```bash
# 1. Start from latest main
git checkout main
git pull

# 2. Create a branch
git checkout -b feature/my-new-agent

# 3. Make changes; commit frequently
git add agents/my_agent.py tests/test_my_agent.py
git commit -m "Add MyAgent for XYZ analysis"

# 4. Run tests before pushing
python3 -m pytest tests/ -v

# 5. Push and open a pull request
git push -u origin feature/my-new-agent
```

### Commit message format

```
<type>: <short description>

<optional body — what and why, not how>
```

Types:

| Type | When to use |
|---|---|
| `feat` | New feature or agent |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change without behaviour change |
| `test` | Add or update tests |
| `chore` | Dependency updates, config changes |

Examples:

```
feat: Add MacroAgent for economic indicator analysis

fix: Handle yfinance MultiIndex for single-ticker downloads

docs: Add screener usage examples to USER_MANUAL.md

test: Add unit tests for RiskAgent stop-loss calculation
```

### Pull request checklist

Before opening a PR:

- [ ] Tests pass: `python3 -m pytest tests/ -v`
- [ ] New logic has tests
- [ ] CHANGELOG.md updated with the change
- [ ] No `.env` file or secrets committed (check `git status`)
- [ ] `requirements.txt` updated if new packages added

---

## 9. Code Conventions

### Python style

- **Python 3.9+** with `from __future__ import annotations` for all new files
- **Type hints** on all function signatures
- **Docstrings** on all public methods (Google or NumPy style)
- Line length: 100 characters

### Module-level logger

Every module should define its logger at the top:

```python
import logging
log = logging.getLogger(__name__)
```

Use the appropriate level:

```python
log.debug("Per-item detail (not shown in terminal by default)")
log.info("Normal progress milestones")
log.warning("Non-fatal issue; operation continues")
log.error("Fatal issue for this item; skip and continue")
```

### Error handling in agents

Agents should not raise exceptions — they should return a safe fallback:

```python
def run(self, ticker: str) -> dict:
    try:
        # ... main logic ...
        return {"signal": signal, "ticker": ticker}
    except Exception as exc:
        log.error("MyAgent failed for %s: %s", ticker, exc)
        return {"signal": "HOLD", "ticker": ticker, "error": str(exc)}
```

The Coordinator handles the case where an agent returns `"HOLD"` due to an error gracefully.

### Configuration constants

All tuneable values belong in `config/settings.py`, not hardcoded in agents:

```python
# config/settings.py
MY_AGENT_THRESHOLD = float(os.getenv("MY_AGENT_THRESHOLD", "0.5"))
```

### Database methods

- All DB methods accept and return plain Python types (no SQLite Row objects)
- Use `dict(row)` when returning query results
- Use `INSERT OR IGNORE` when a `UNIQUE` constraint exists
- Always include a `created_at TEXT NOT NULL` column in new tables

### External API calls

- Always set explicit timeouts: `requests.get(url, timeout=10)`
- Always call `resp.raise_for_status()` before parsing the response
- Suppress third-party library logs at the module level when they produce false alarms (see `screener_agent.py` for the yfinance example)
