"""
Configuration settings for the News Trading System.

Loads environment variables via python-dotenv and exposes system-wide
constants so that every module imports from a single source of truth.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------

NEWSAPI_KEY: str = os.environ.get("NEWSAPI_KEY", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
REDDIT_CLIENT_ID: str = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.environ.get("REDDIT_USER_AGENT", "news-trading-bot/1.0")
MARKETAUX_API_TOKEN: str = os.environ.get("MARKETAUX_API_TOKEN", "")
ADANOS_API_KEY: str = os.environ.get("ADANOS_API_KEY", "")
FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")
EODHD_API_TOKEN: str = os.environ.get("EODHD_API_TOKEN", "")
POLYGON_API_KEY: str = os.environ.get("POLYGON_API_KEY", "")

# ---------------------------------------------------------------------------
# Claude model
# ---------------------------------------------------------------------------

CLAUDE_MODEL: str = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------

NEWSAPI_URL: str = "https://newsapi.org/v2/everything"
MAX_HEADLINES: int = 10

# ---------------------------------------------------------------------------
# Sentiment → numeric score mapping
# ---------------------------------------------------------------------------

SCORE_MAP: dict[str, int] = {"bullish": 1, "neutral": 0, "bearish": -1}

# ---------------------------------------------------------------------------
# Trading signal thresholds
# ---------------------------------------------------------------------------

BUY_THRESHOLD: float = 0.3
SELL_THRESHOLD: float = -0.3

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------

DRAWDOWN_HALT_THRESHOLD: float = float(
    os.environ.get("DRAWDOWN_HALT_THRESHOLD", "0.10")
)

# ---------------------------------------------------------------------------
# Source weights for multi-source sentiment
# ---------------------------------------------------------------------------

SOURCE_WEIGHTS: dict[str, float] = {
    "newsapi": 1.0,
    "marketaux": 0.9,
    "stocktwits": 0.8,
    "reddit": 0.6,
    "adanos": 0.5,
    "apewisdom": 0.4,
}

# When Reddit is unavailable, redistribute its weight to NewsAPI and Marketaux
# so the remaining sources carry the same total influence.
SOURCE_WEIGHTS_NO_REDDIT: dict[str, float] = {
    "newsapi": 1.2,
    "marketaux": 1.1,
    "stocktwits": 0.8,
    "adanos": 0.5,
    "apewisdom": 0.4,
}

# When NewsAPI is unavailable (rate-limited / 429), redistribute its weight
# to Marketaux and StockTwits so the remaining sources carry similar influence.
SOURCE_WEIGHTS_NO_NEWSAPI: dict[str, float] = {
    "marketaux": 1.2,
    "stocktwits": 1.0,
    "reddit": 0.7,
    "adanos": 0.5,
    "apewisdom": 0.5,
}

# When both Reddit and NewsAPI are unavailable.
SOURCE_WEIGHTS_NO_REDDIT_NO_NEWSAPI: dict[str, float] = {
    "marketaux": 1.3,
    "stocktwits": 1.1,
    "adanos": 0.5,
    "apewisdom": 0.5,
}

# Known crypto tickers (used to route to Binance instead of yfinance)
CRYPTO_TICKERS: set[str] = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX",
    "MATIC", "LINK", "UNI", "ATOM", "LTC", "FIL", "NEAR", "APT", "ARB",
}

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

DB_PATH: str = os.environ.get("DB_PATH", "news_trading.db")

# ---------------------------------------------------------------------------
# Daily OHLC ingest (Research-side; scripts/ingest_ohlc.py → daily_ohlc table)
# ---------------------------------------------------------------------------

# Daily-bar source for scripts/ingest_ohlc.py: "alpaca" (default since 2026-09-08,
# free Basic plan, SIP daily bars, 200 req/min) or "polygon" (rollback: free tier,
# 5 req/min, 2yr history). Decision + bar comparison: docs/OHLC_SOURCE_SWITCH_2026-09-08.md
OHLC_SOURCE: str = os.environ.get("OHLC_SOURCE", "alpaca")
OHLC_BACKFILL_YEARS: int = 2          # both free tiers cover >= 2yr daily; covers 252d windows
OHLC_EXTREME_MOVE_PCT: float = 0.50   # soft-flag threshold (quality_flag='EXTREME_MOVE'), NOT a reject

# ---------------------------------------------------------------------------
# IBKR (Interactive Brokers) — used when TRADING_MODE=ibkr_paper or ibkr_live
# ---------------------------------------------------------------------------

IBKR_HOST: str = os.environ.get("IBKR_HOST", "127.0.0.1")
IBKR_PORT: int = int(os.environ.get("IBKR_PORT", "0"))  # 0 = auto (4002 paper / 4001 live)
IBKR_CLIENT_ID: int = int(os.environ.get("IBKR_CLIENT_ID", "1"))
IBKR_PAPER: bool = os.environ.get("IBKR_PAPER", "true").lower() in ("true", "1", "yes")
USE_IBKR_DATA: bool = os.environ.get("USE_IBKR_DATA", "false").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# German / EU ticker lists (used to route to EODHD)
# ---------------------------------------------------------------------------

DAX_TICKERS: list[str] = [
    "SAP.XETRA", "SIE.XETRA", "ALV.XETRA", "MUV2.XETRA", "BMW.XETRA",
    "VOW3.XETRA", "MBG.XETRA", "DTE.XETRA", "BAYN.XETRA", "BAS.XETRA",
    "ADS.XETRA", "RWE.XETRA", "EOAN.XETRA", "DBK.XETRA", "IFX.XETRA",
    "DHL.XETRA", "DB1.XETRA", "LIN.XETRA", "MRK.XETRA", "HEI.XETRA",
    "HEN3.XETRA", "FRE.XETRA", "ZAL.XETRA", "CON.XETRA", "VNA.XETRA",
    "RHM.XETRA", "AIR.XETRA", "PAH3.XETRA", "P911.XETRA", "BNR.XETRA",
    "MTX.XETRA", "SRT3.XETRA", "DHER.XETRA", "FME.XETRA", "CBK.XETRA",
    "HNR1.XETRA", "ENR.XETRA", "SHL.XETRA", "EVK.XETRA", "SY1.XETRA",
]

MDAX_TICKERS: list[str] = [
    "AFX.XETRA", "AIXA.XETRA", "BC8.XETRA", "BOSS.XETRA", "DWS.XETRA",
    "EVD.XETRA", "FNTN.XETRA", "GBF.XETRA", "HOT.XETRA", "LEG.XETRA",
    "MLP.XETRA", "NDX1.XETRA", "PSM.XETRA", "PUM.XETRA", "QIA.XETRA",
    "TAG.XETRA", "TUI1.XETRA", "WCHA.XETRA", "O2D.XETRA", "SDF.XETRA",
    "RDC.XETRA", "VBK.XETRA", "LHA.XETRA", "TKA.XETRA", "UTDI.XETRA",
]

# All known German tickers (union of DAX + MDAX)
_GERMAN_TICKERS_SET: set[str] = set(DAX_TICKERS) | set(MDAX_TICKERS)

# Ticker → company name for NewsAPI search (searching "SAP.XETRA" returns junk)
TICKER_TO_COMPANY: dict[str, str] = {
    # DAX 40
    "SAP.XETRA": "SAP SE", "SAP.DE": "SAP SE",
    "SIE.XETRA": "Siemens AG", "SIE.DE": "Siemens AG",
    "ALV.XETRA": "Allianz SE", "ALV.DE": "Allianz SE",
    "MUV2.XETRA": "Munich Re", "MUV2.DE": "Munich Re",
    "BMW.XETRA": "BMW AG", "BMW.DE": "BMW AG",
    "VOW3.XETRA": "Volkswagen AG", "VOW3.DE": "Volkswagen AG",
    "MBG.XETRA": "Mercedes-Benz Group", "MBG.DE": "Mercedes-Benz Group",
    "DTE.XETRA": "Deutsche Telekom", "DTE.DE": "Deutsche Telekom",
    "BAYN.XETRA": "Bayer AG", "BAYN.DE": "Bayer AG",
    "BAS.XETRA": "BASF SE", "BAS.DE": "BASF SE",
    "ADS.XETRA": "Adidas AG", "ADS.DE": "Adidas AG",
    "RWE.XETRA": "RWE AG", "RWE.DE": "RWE AG",
    "EOAN.XETRA": "E.ON SE", "EOAN.DE": "E.ON SE",
    "DBK.XETRA": "Deutsche Bank", "DBK.DE": "Deutsche Bank",
    "IFX.XETRA": "Infineon Technologies", "IFX.DE": "Infineon Technologies",
    "DHL.XETRA": "DHL Group", "DHL.DE": "DHL Group",
    "DB1.XETRA": "Deutsche Boerse", "DB1.DE": "Deutsche Boerse",
    "LIN.XETRA": "Linde plc", "LIN.DE": "Linde plc",
    "MRK.XETRA": "Merck KGaA", "MRK.DE": "Merck KGaA",
    "HEI.XETRA": "HeidelbergCement", "HEI.DE": "HeidelbergCement",
    "HEN3.XETRA": "Henkel AG", "HEN3.DE": "Henkel AG",
    "FRE.XETRA": "Fresenius SE", "FRE.DE": "Fresenius SE",
    "ZAL.XETRA": "Zalando SE", "ZAL.DE": "Zalando SE",
    "CON.XETRA": "Continental AG", "CON.DE": "Continental AG",
    "VNA.XETRA": "Vonovia SE", "VNA.DE": "Vonovia SE",
    "RHM.XETRA": "Rheinmetall AG", "RHM.DE": "Rheinmetall AG",
    "AIR.XETRA": "Airbus SE", "AIR.DE": "Airbus SE",
    "PAH3.XETRA": "Porsche Automobil Holding", "PAH3.DE": "Porsche Automobil Holding",
    "P911.XETRA": "Porsche AG", "P911.DE": "Porsche AG",
    "BNR.XETRA": "Brenntag SE", "BNR.DE": "Brenntag SE",
    "MTX.XETRA": "MTU Aero Engines", "MTX.DE": "MTU Aero Engines",
    "SRT3.XETRA": "Sartorius AG", "SRT3.DE": "Sartorius AG",
    "DHER.XETRA": "Delivery Hero", "DHER.DE": "Delivery Hero",
    "FME.XETRA": "Fresenius Medical Care", "FME.DE": "Fresenius Medical Care",
    "CBK.XETRA": "Commerzbank AG", "CBK.DE": "Commerzbank AG",
    "HNR1.XETRA": "Hannover Rueck", "HNR1.DE": "Hannover Rueck",
    "ENR.XETRA": "Siemens Energy", "ENR.DE": "Siemens Energy",
    "SHL.XETRA": "Siemens Healthineers", "SHL.DE": "Siemens Healthineers",
    "EVK.XETRA": "Evonik Industries", "EVK.DE": "Evonik Industries",
    "SY1.XETRA": "Symrise AG", "SY1.DE": "Symrise AG",
}

# Adanos — disabled by default (free tier quota too small)
ADANOS_ENABLED: bool = os.environ.get("ADANOS_ENABLED", "false").lower() in ("true", "1", "yes")

# Pre-market sessions (XETRA_PRE, US_PRE) — lightweight signal refresh
ENABLE_PRE_SESSIONS: bool = os.environ.get("ENABLE_PRE_SESSIONS", "true").lower() in ("true", "1", "yes")

# Per-ticker regime detection (ADX/VIX/ATR) — controls strategy activation + sizing
ENABLE_REGIME_FILTER: bool = os.environ.get("ENABLE_REGIME_FILTER", "true").lower() in ("true", "1", "yes")

# Prompt caching — reduces Anthropic API costs 60-80% on repeated system prompts
ENABLE_PROMPT_CACHING: bool = os.environ.get("ENABLE_PROMPT_CACHING", "true").lower() in ("true", "1", "yes")

# PEAD (Post-Earnings Announcement Drift) strategy
PEAD_ENABLED: bool = os.environ.get("PEAD_ENABLED", "true").lower() in ("true", "1", "yes")
PEAD_TICKERS: list[str] = [
    # ticker selection: no artifact. Aggregate DSR 1.00 in
    # combined_pead_ibkr_final.json (5ac2488) covers the 30-ticker
    # US leg, not this 9-name subset; the production-config rerun
    # 74a1154 (2026-03-31) reported DSR 0.0000. See KB-1 P7 audit.
    "CASY", "TXRH", "DECK", "TRGP", "CACI", "MEDP", "UFPI", "TOL", "PBR",
    # EU mid-caps (top Sharpe from IBKR cache run)
    "VNA.DE", "HOT.DE", "COFA.PA", "VIE.PA", "FNTN.DE", "LEG.DE",
]
# ATR-based dynamic stop-loss / take-profit
USE_ATR_STOPS: bool = os.environ.get("USE_ATR_STOPS", "true").lower() in ("true", "1", "yes")
ATR_STOP_MULTIPLIER: float = float(os.environ.get("ATR_STOP_MULTIPLIER", "1.5"))
ATR_TP_MULTIPLIER: float = float(os.environ.get("ATR_TP_MULTIPLIER", "3.0"))
ACCOUNT_RISK_PCT: float = float(os.environ.get("ACCOUNT_RISK_PCT", "0.01"))

# ---------------------------------------------------------------------------
# Cluster agreement gate (2026-09-07) — the one place these live.
# docs/CLUSTER_AGREEMENT_GATE_2026-09-07.md
# ---------------------------------------------------------------------------
# ClusterDetector emits a directional verdict only when at least this many
# *distinct vote sources* point the same way (each vote ≥ MIN_CONFIDENCE).
# Below the threshold the verdict is HOLD and the run is logged to
# signal_events with cluster_gate='rejected_min_agreement' plus the vote
# count, direction and voters.  1 restores the pre-gate behaviour (a solo
# vote passes).  Not an env override on purpose: the value is a trading
# rule, not a deployment knob.
CLUSTER_MIN_AGREEING_STRATEGIES: int = 2

# Strategy name → vote source.  Strategies mapped to the same source count
# ONCE toward the threshold (two votes fed by one input stream are not
# convergence).  Default: every strategy is its own source.  Momentum and
# NewsCatalyst share the sentiment feed and the 20-day volume ratio (see
# the doc, §2 — they co-fire 3.4× more often than independence predicts);
# mapping both to one source is the knob for folding them.  Left
# unfolded here — that is a decision, not a default.
CLUSTER_VOTE_SOURCES: dict[str, str] = {
    "Momentum": "Momentum",
    "Pullback": "Pullback",
    "NewsCatalyst": "NewsCatalyst",
}

# ---------------------------------------------------------------------------
# Re-entry lock after a stop-loss exit (PortfolioManager gate 1b)
# docs/REENTRY_LOCK_2026-09-08.md
# ---------------------------------------------------------------------------
# After a round trip ends in a stop-loss exit, the same ticker cannot be
# bought again for this many US trading sessions (data.market_calendar —
# weekends AND full-day holidays do not count as sessions). 0 = same session
# only, 1 = the pre-2026-09 Q-016 cool-down. Every rejected BUY is written to
# signal_events (strategy='PortfolioManager', signal_path='REENTRY_LOCK')
# with the stop timestamp, the sessions elapsed / remaining and the first
# eligible session. 5 = one trading week: in the audited book (2026-05-04 …
# 2026-09-04, 78 after-stop re-entries) 62 % of them landed within 4
# sessions, sessions 5–6 were empty, and the tail beyond is diffuse — the
# lock covers the dense cluster and ends at the empirical break. Not an env
# override on purpose: a trading rule, not a deployment knob.
REENTRY_LOCK_SESSIONS: int = 5

# Whether a TRAILING-stop exit (PositionManager 'stop_loss_triggered' at a
# level above the entry stop, i.e. a profitable exit after a >2 % run-up)
# also arms the lock. Off: the lock covers stop-LOSS exits only, which is
# what the rule says. The audit found re-entries after trailing exits
# equally poor (N=47, −1.00 % per trade) — that is reported, not folded in.
REENTRY_LOCK_INCLUDE_TRAILING: bool = False

PEAD_EARNINGS_CACHE_PATH: str = os.environ.get(
    "PEAD_EARNINGS_CACHE_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "walk-forward-backtest" / "data" / "ibkr_earnings_cache.json"),
)


# ---------------------------------------------------------------------------
# Optimized trend-following parameters (2-stage walk-forward, 2023–2025)
# 9 production-ready tickers; 7 skipped (overfit / low trades / negative OOS)
# ---------------------------------------------------------------------------

TREND_PARAMS: dict[str, dict] = {
    # --- AI_CHIPS (sector consensus: SMA 10/200, SL 1.0%, TP 2.25x) ---
    "MSFT":  {"sma_fast": 10, "sma_slow": 200, "stop_loss_pct": 0.010, "take_profit_ratio": 2.25},
    "META":  {"sma_fast": 50, "sma_slow": 100, "stop_loss_pct": 0.020, "take_profit_ratio": 2.75},
    "GOOGL": {"sma_fast": 20, "sma_slow": 200, "stop_loss_pct": 0.015, "take_profit_ratio": 2.50},
    # --- DATACENTER (sector consensus: SMA 20/200, SL 1.5%, TP 2.5x) ---
    "VST":   {"sma_fast": 20, "sma_slow": 200, "stop_loss_pct": 0.015, "take_profit_ratio": 2.50},
    "CEG":   {"sma_fast": 50, "sma_slow": 200, "stop_loss_pct": 0.030, "take_profit_ratio": 2.00},
    "AAPL":  {"sma_fast": 30, "sma_slow":  75, "stop_loss_pct": 0.015, "take_profit_ratio": 2.25},
    "DELL":  {"sma_fast": 20, "sma_slow": 200, "stop_loss_pct": 0.015, "take_profit_ratio": 2.50},
    # --- GERMAN_TECH ---
    "SAP.XETRA": {"sma_fast": 10, "sma_slow": 175, "stop_loss_pct": 0.035, "take_profit_ratio": 2.50},
    # --- CRYPTO ---
    "SOL":   {"sma_fast": 20, "sma_slow": 125, "stop_loss_pct": 0.035, "take_profit_ratio": 2.50},
}

# Shared defaults for all trend tickers (RSI fixed at standard values)
TREND_DEFAULTS: dict[str, object] = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "require_volume_confirmation": False,
    "use_sentiment": False,
    "use_technical": True,
    "require_trend_alignment": True,
}

# Tickers skipped (overfit/low trades): NVDA, AMD, TSLA, SMCI, SIE.XETRA, BTC, ETH


def is_german_ticker(ticker: str) -> bool:
    """Return True if *ticker* should be routed to EODHD (German/EU stock)."""
    t = ticker.upper()
    if t.endswith(".XETRA") or t.endswith(".DE"):
        return True
    return t in _GERMAN_TICKERS_SET


def get_search_term(ticker: str) -> str:
    """Return the best NewsAPI search term for *ticker* (company name for German stocks)."""
    return TICKER_TO_COMPANY.get(ticker.upper(), ticker)


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_api_keys() -> None:
    """Raise RuntimeError if required API keys are missing."""
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not NEWSAPI_KEY:
        missing.append("NEWSAPI_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in your .env file or export them in your shell."
        )
