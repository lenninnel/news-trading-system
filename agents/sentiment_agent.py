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
     error) the agent falls back to a rule-based keyword scorer that
     assigns sentiment without any external API call.

  3. Fallback results carry ``"degraded": True`` so callers can optionally
     reduce the weight of rule-based scores in their aggregation logic.

  4. All fallback activations are logged to recovery_log when a Database
     instance has been attached via APIRecovery.set_db(db).

Rule-based fallback keyword lists
----------------------------------
Bullish: beat, record, surge, rise, up, gain, buy, upgrade, strong,
         profit, revenue, growth, bull, outperform, positive, high,
         exceed, raise, expand, accelerate, rally, soar, boom, wins

Bearish: miss, drop, fall, down, loss, cut, downgrade, weak, decline,
         bear, underperform, negative, low, below, risk, concern,
         lawsuit, fraud, fine, bankruptcy, layoff, recall, crash
"""

from __future__ import annotations

import json
import logging
import re

import anthropic

from agents.base_agent import BaseAgent
from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, SCORE_MAP
from utils.api_recovery import APIRecovery, CircuitOpenError, UnauthorizedError

log = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    'You are a financial sentiment analyst. For the stock ticker "{ticker}", '
    "classify the following headline as exactly one of: bullish, bearish, or neutral.\n\n"
    'Headline: "{headline}"\n\n'
    "Respond with ONLY valid JSON (no markdown) in this format:\n"
    '{{"sentiment": "bullish|bearish|neutral", "reason": "<one sentence>"}}'
)

# ── Rule-based keyword sets ────────────────────────────────────────────────────

_BULLISH_WORDS = frozenset(
    "beat record surge rise up gain buy upgrade strong profit revenue growth "
    "bull outperform positive high exceed raise expand accelerate rally soar "
    "boom launch wins award breakthrough partnership deal acquisition".split()
)

_BEARISH_WORDS = frozenset(
    "miss drop fall down loss cut downgrade weak decline bear underperform "
    "negative low below risk concern lawsuit fraud fine bankruptcy layoff "
    "recall crash collapse plunge tumble warn delay halt suspend "
    "investigate probe penalty breach hack short sell".split()
)


def _rule_based_sentiment(headline: str) -> dict:
    """
    Keyword-based sentiment scorer — no external API required.

    Tokenises the headline, counts matching bullish/bearish keywords, and
    returns a result dict in the same format as the Claude-powered scorer.

    Returns:
        dict with: sentiment, score, reason, headline, degraded=True
    """
    words   = re.findall(r"[a-z]+", headline.lower())
    bullish = sum(1 for w in words if w in _BULLISH_WORDS)
    bearish = sum(1 for w in words if w in _BEARISH_WORDS)

    if bullish > bearish:
        sentiment = "bullish"
        reason    = f"Rule-based: {bullish} bullish keyword(s) detected"
    elif bearish > bullish:
        sentiment = "bearish"
        reason    = f"Rule-based: {bearish} bearish keyword(s) detected"
    else:
        sentiment = "neutral"
        reason    = "Rule-based: balanced or no sentiment keywords found"

    return {
        "sentiment": sentiment,
        "score":     SCORE_MAP.get(sentiment, 0),
        "reason":    reason,
        "headline":  headline,
        "degraded":  True,
    }


class SentimentAgent(BaseAgent):
    """
    Classifies news headlines using Claude; falls back to keyword scoring.

    Attributes:
        _client: Shared Anthropic client instance.

    Example::

        agent = SentimentAgent()
        result = agent.run("Apple hits all-time high", "AAPL")
        # {"sentiment": "bullish", "score": 1, "reason": "...", "headline": "...", "degraded": False}
        # On Anthropic failure:
        # {"sentiment": "bullish", "score": 1, "reason": "Rule-based: ...", "degraded": True}
    """

    def __init__(self, api_key: str = ANTHROPIC_API_KEY) -> None:
        """
        Initialise the agent with an Anthropic client.

        Args:
            api_key: Anthropic API key. Defaults to settings.ANTHROPIC_API_KEY.
        """
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "SentimentAgent"

    def run(self, headline: str, ticker: str) -> dict:
        """
        Classify a single headline for the given ticker.

        Attempts Claude API first; falls back to keyword scoring on failure.

        Args:
            headline: The news headline text to analyse.
            ticker:   Stock ticker symbol (e.g. "AAPL").

        Returns:
            dict with keys:
                sentiment (str):  "bullish", "bearish", or "neutral"
                score     (int):  +1, 0, or -1
                reason    (str):  Justification string
                headline  (str):  Original headline (pass-through)
                degraded  (bool): True when rule-based fallback was used
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
                "Using rule-based fallback."
            )
            self._log_degraded(ticker, "HTTP 401 Unauthorized")
            return _rule_based_sentiment(headline)
        except CircuitOpenError as exc:
            log.warning(
                "[DEGRADED MODE] Anthropic circuit OPEN for %s — rule-based fallback",
                ticker,
            )
            self._log_degraded(ticker, str(exc))
            return _rule_based_sentiment(headline)
        except Exception as exc:
            log.warning(
                "[DEGRADED MODE] Anthropic unavailable for %s (%s) — rule-based fallback",
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
        result["score"]    = SCORE_MAP.get(result["sentiment"], 0)
        result["headline"] = headline
        result["degraded"] = False
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
                recovery_action="rule_based_sentiment_fallback",
                success=False,
            )
        except Exception:
            pass
