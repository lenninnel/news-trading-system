"""
Sentiment analysis agent powered by Claude.

SentimentAgent receives a single news headline and a ticker symbol, asks
Claude to classify the headline as bullish, bearish, or neutral, and returns
a structured result dict.

Recovery behaviour
------------------
  1. APIRecovery.call("anthropic", …) wraps every Claude API call with
     per-service retry logic (max 2 attempts, 30 s backoff) and a circuit
     breaker (opens after 5 consecutive failures).

  2. On Anthropic failure (circuit open, all retries exhausted, or auth
     error) the agent falls back to a rule-based lexicon scorer.

  3. Fallback results carry ``"degraded": True`` and a lower confidence
     score (0.55 vs 0.85 for Claude) so callers can adjust signal weights.

  4. All fallback activations are logged to recovery_log when a Database
     instance has been attached via APIRecovery.set_db(db).

Confidence scores
-----------------
  Claude-based   : 0.85  (high accuracy, occasional hallucinations)
  Rule-based     : 0.55  (lexicon-weighted, less context-aware)

Sentiment lexicon
-----------------
  Loaded at module startup from ``data/sentiment_lexicon.json``.
  Falls back to a smaller hardcoded set if the file is missing.
  Supports:
    • Per-term weights (1.0 = normal, 1.5 = strong, 2.0 = very strong)
    • Amplifier words (multiply adjacent term weight by 1.5)
    • Negator words (invert the sentiment of the next term)
    • Compound phrases (multi-word patterns matched before word-level scoring)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import anthropic

from agents.base_agent import BaseAgent
from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, SCORE_MAP
from utils.api_recovery import APIRecovery, CircuitOpenError, UnauthorizedError

log = logging.getLogger(__name__)

_LEXICON_PATH = Path(__file__).resolve().parent.parent / "data" / "sentiment_lexicon.json"

_PROMPT_TEMPLATE = (
    'You are a financial sentiment analyst. For the stock ticker "{ticker}", '
    "classify the following headline as exactly one of: bullish, bearish, or neutral.\n\n"
    'Headline: "{headline}"\n\n'
    "Respond with ONLY valid JSON (no markdown) in this format:\n"
    '{{"sentiment": "bullish|bearish|neutral", "reason": "<one sentence>"}}'
)

# Confidence constants
_CONFIDENCE_CLAUDE     = 0.85
_CONFIDENCE_RULE_BASED = 0.55


# ── Lexicon loading ────────────────────────────────────────────────────────────

def _load_lexicon() -> dict:
    """
    Load sentiment lexicon from JSON file.

    Returns a dict with keys:
        bullish_weights  : dict[str, float]
        bearish_weights  : dict[str, float]
        amplifiers       : frozenset[str]
        negators         : frozenset[str]
        compound_bullish : list[str]
        compound_bearish : list[str]
    Falls back to minimal hardcoded sets on any error.
    """
    try:
        raw = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
        return {
            "bullish_weights":  {e["term"]: e["weight"] for e in raw.get("bullish", [])},
            "bearish_weights":  {e["term"]: e["weight"] for e in raw.get("bearish", [])},
            "amplifiers":       frozenset(raw.get("amplifiers", [])),
            "negators":         frozenset(raw.get("negators", [])),
            "compound_bullish": [p.lower() for p in raw.get("compound_bullish", [])],
            "compound_bearish": [p.lower() for p in raw.get("compound_bearish", [])],
        }
    except Exception as exc:
        log.warning("Could not load sentiment_lexicon.json (%s) — using hardcoded fallback", exc)
        return _hardcoded_lexicon()


def _hardcoded_lexicon() -> dict:
    """Minimal hardcoded lexicon used when the JSON file is unavailable."""
    bullish = {
        "beat": 1.2, "record": 1.5, "profit": 1.2, "revenue": 1.0, "growth": 1.2,
        "surge": 1.5, "soar": 1.5, "rally": 1.3, "gain": 1.0, "rise": 1.0,
        "buy": 1.0, "upgrade": 1.3, "outperform": 1.3, "strong": 1.0,
        "positive": 1.0, "exceed": 1.2, "boom": 1.3, "breakthrough": 1.5,
        "wins": 1.2, "recovery": 1.2, "bullish": 1.3, "expand": 1.0,
    }
    bearish = {
        "miss": 1.2, "drop": 1.2, "fall": 1.0, "loss": 1.2, "cut": 1.2,
        "downgrade": 1.3, "weak": 1.0, "decline": 1.2, "bearish": 1.3,
        "underperform": 1.3, "negative": 1.0, "concern": 1.0,
        "lawsuit": 1.5, "fraud": 2.0, "bankruptcy": 2.0, "layoff": 1.3,
        "crash": 1.5, "collapse": 1.5, "plunge": 1.5, "warn": 1.2,
        "investigation": 1.5, "penalty": 1.3, "default": 2.0,
    }
    return {
        "bullish_weights":  bullish,
        "bearish_weights":  bearish,
        "amplifiers":       frozenset(["record", "massive", "unprecedented", "significant"]),
        "negators":         frozenset(["not", "no", "never", "without", "fails"]),
        "compound_bullish": ["beats expectations", "record profit", "raises guidance"],
        "compound_bearish": ["misses expectations", "faces bankruptcy", "sec investigation"],
    }


# Load at module import (cached for the lifetime of the process)
_LEXICON: dict = _load_lexicon()


# ── Rule-based scorer ─────────────────────────────────────────────────────────

def _rule_based_sentiment(headline: str) -> dict:
    """
    Production-grade lexicon-based sentiment scorer.

    Algorithm
    ---------
    1. Check compound phrases first (multi-word patterns, highest priority).
    2. Tokenise headline into lowercase words.
    3. For each word:
       a. If it's a negator → flip the sentiment direction for the next term.
       b. If it's an amplifier → boost the next term's weight by ×1.5.
       c. If it's in bullish/bearish dict → add weighted score.
    4. Net score > 0 → bullish, < 0 → bearish, = 0 → neutral.

    Returns:
        dict with: sentiment, score, reason, headline, degraded=True,
                   confidence=_CONFIDENCE_RULE_BASED
    """
    text  = headline.lower()
    words = re.findall(r"[a-z]+", text)

    lex      = _LEXICON
    bw       = lex["bullish_weights"]
    sw       = lex["bearish_weights"]
    amps     = lex["amplifiers"]
    negs     = lex["negators"]
    c_bull   = lex["compound_bullish"]
    c_bear   = lex["compound_bearish"]

    # 1. Compound phrase scan
    compound_score = 0.0
    for phrase in c_bull:
        if phrase in text:
            compound_score += 2.0
    for phrase in c_bear:
        if phrase in text:
            compound_score -= 2.0

    # 2. Word-level weighted scan
    word_score   = 0.0
    next_amplify = 1.0
    next_negate  = False

    for w in words:
        if w in negs:
            next_negate  = True
            next_amplify = 1.0
            continue
        if w in amps:
            next_amplify = 1.5
            continue

        weight = 0.0
        if w in bw:
            weight = +bw[w]
        elif w in sw:
            weight = -sw[w]

        if weight != 0.0:
            if next_negate:
                weight = -weight
            word_score += weight * next_amplify

        # Reset modifiers after a scored term
        if w in bw or w in sw:
            next_amplify = 1.0
            next_negate  = False

    total = compound_score + word_score

    if total > 0.3:
        sentiment = "bullish"
        reason    = f"Lexicon: net score {total:+.2f} (bullish keywords outweigh bearish)"
    elif total < -0.3:
        sentiment = "bearish"
        reason    = f"Lexicon: net score {total:+.2f} (bearish keywords outweigh bullish)"
    else:
        sentiment = "neutral"
        reason    = f"Lexicon: net score {total:+.2f} (balanced or no strong keywords)"

    return {
        "sentiment":  sentiment,
        "score":      SCORE_MAP.get(sentiment, 0),
        "reason":     reason,
        "headline":   headline,
        "degraded":   True,
        "confidence": _CONFIDENCE_RULE_BASED,
    }


class SentimentAgent(BaseAgent):
    """
    Classifies news headlines using Claude; falls back to lexicon scoring.

    Example::

        agent = SentimentAgent()
        result = agent.run("Apple hits all-time high", "AAPL")
        # {"sentiment": "bullish", "score": 1, "reason": "...",
        #  "headline": "...", "degraded": False, "confidence": 0.85}
        # On Anthropic failure:
        # {"sentiment": "bullish", "score": 1, "reason": "Lexicon: ...",
        #  "degraded": True, "confidence": 0.55}
    """

    def __init__(self, api_key: str = ANTHROPIC_API_KEY) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "SentimentAgent"

    def run(self, headline: str, ticker: str) -> dict:
        """
        Classify a single headline for the given ticker.

        Attempts Claude API first; falls back to lexicon scoring on failure.

        Returns:
            dict with: sentiment, score, reason, headline, degraded, confidence
        """
        try:
            return APIRecovery.call(
                "anthropic",
                self._claude_classify,
                headline,
                ticker,
                ticker=ticker,
            )
        except UnauthorizedError:
            log.error(
                "Anthropic API key rejected — check ANTHROPIC_API_KEY. "
                "Using lexicon fallback."
            )
            self._log_degraded(ticker, "HTTP 401 Unauthorized")
            return _rule_based_sentiment(headline)
        except CircuitOpenError as exc:
            log.warning(
                "[DEGRADED MODE] Anthropic circuit OPEN for %s — lexicon fallback", ticker
            )
            self._log_degraded(ticker, str(exc))
            return _rule_based_sentiment(headline)
        except Exception as exc:
            log.warning(
                "[DEGRADED MODE] Anthropic unavailable for %s (%s) — lexicon fallback",
                ticker, exc,
            )
            self._log_degraded(ticker, str(exc))
            return _rule_based_sentiment(headline)

    def _claude_classify(self, headline: str, ticker: str) -> dict:
        """Raw Claude API call; raises on any error."""
        prompt = _PROMPT_TEMPLATE.format(ticker=ticker, headline=headline)
        message = self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text: str    = message.content[0].text.strip()
        result: dict = json.loads(text)
        result["score"]      = SCORE_MAP.get(result["sentiment"], 0)
        result["headline"]   = headline
        result["degraded"]   = False
        result["confidence"] = _CONFIDENCE_CLAUDE
        return result

    @staticmethod
    def _log_degraded(ticker: str, error: str) -> None:
        db = APIRecovery._db
        if db is None:
            return
        try:
            db.log_recovery_event(
                service="anthropic",
                event_type="degraded_mode",
                ticker=ticker,
                error_msg=error,
                recovery_action="lexicon_sentiment_fallback",
                success=False,
            )
        except Exception:
            pass
