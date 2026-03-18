"""
Configuration settings for the News Trading System.

Loads environment variables via python-dotenv and exposes system-wide
constants so that every module imports from a single source of truth.
"""

import os

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
