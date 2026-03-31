"""
Bull / Bear Debate agent — adversarial reasoning before trade decisions.

Inspired by TradingAgents (github.com/TauricResearch/TradingAgents).

Two virtual analysts argue opposing sides of every non-HOLD signal:

    BullResearcher — argues FOR the trade, finds supporting evidence
    BearResearcher — argues AGAINST the trade, surfaces risks

A synthesis step merges both perspectives into a final verdict with an
adjusted confidence score.

When both researchers *agree* on direction, confidence gets a boost.
When they *disagree*, confidence is penalised and the system becomes
more cautious.

Recovery behaviour
------------------
If the debate fails for any reason (API error, timeout, malformed
response), the original signal and confidence pass through unchanged
so the trading pipeline is never blocked.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import re

import anthropic

from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, ENABLE_PROMPT_CACHING
from utils.api_recovery import APIRecovery

log = logging.getLogger(__name__)

# ── Result container ─────────────────────────────────────────────────────────


@dataclass
class DebateResult:
    """Immutable container for the outcome of a bull/bear debate."""

    ticker: str
    original_signal: str
    original_confidence: float
    bull_case: str = ""
    bear_case: str = ""
    final_signal: str = ""
    adjusted_confidence: float = 0.0
    debate_adjustment: float = 0.0
    debate_summary: str = ""
    degraded: bool = False


# ── Prompt templates ─────────────────────────────────────────────────────────

# Static system prompts (cached across calls — same for every ticker)
_BULL_SYSTEM = (
    "You are a bullish equity analyst making the strongest possible case "
    "FOR taking a trade position. Argue convincingly why the trade SHOULD "
    "be taken. Cite the data provided.\n\n"
    "Respond with ONLY valid JSON:\n"
    '{"bull_case": "<2-3 sentences>", "confidence_boost": <float -0.2 to 0.2>}'
)

_BEAR_SYSTEM = (
    "You are a skeptical risk manager making the strongest possible case "
    "AGAINST taking a trade position. Argue convincingly why the trade is "
    "RISKY and should NOT be taken. Identify weaknesses, contrary "
    "indicators, and potential traps.\n\n"
    "Respond with ONLY valid JSON:\n"
    '{"bear_case": "<2-3 sentences>", "confidence_penalty": <float -0.2 to 0.0>}'
)

# Dynamic user message template (changes per ticker)
_BULL_USER = (
    "Ticker: {ticker}\n"
    "Signal: {signal}\n"
    "Confidence: {confidence:.0%}\n"
    "Technical indicators: {technical}\n"
    "Sentiment summary: {sentiment}"
)

_BEAR_USER = _BULL_USER  # same data, different system prompt role


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON output from Claude."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _log_cache_usage(msg: Any, caller: str) -> None:
    """Log prompt cache hit/miss from Anthropic response usage."""
    try:
        usage = getattr(msg, "usage", None)
        if usage is None:
            return
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_create = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        if cache_read > 0:
            log.debug("%s cache HIT: %d tokens read from cache", caller, cache_read)
        elif cache_create > 0:
            log.debug("%s cache WRITE: %d tokens cached", caller, cache_create)
    except (TypeError, ValueError):
        pass  # mock objects or unexpected types — skip logging


# ── Individual researchers ───────────────────────────────────────────────────


class BullResearcher:
    """Argues FOR the trade. Finds supporting evidence."""

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def analyze(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        technical_data: dict,
        sentiment_data: dict,
    ) -> dict:
        user_msg = _BULL_USER.format(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            technical=json.dumps(technical_data, default=str),
            sentiment=json.dumps(sentiment_data, default=str),
        )

        def _call():
            kwargs: dict = {
                "model": CLAUDE_MODEL,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": user_msg}],
            }
            if ENABLE_PROMPT_CACHING:
                kwargs["system"] = [{
                    "type": "text",
                    "text": _BULL_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                kwargs["system"] = _BULL_SYSTEM

            msg = self._client.messages.create(**kwargs)
            _log_cache_usage(msg, "BullResearcher")
            return json.loads(_strip_fences(msg.content[0].text))

        try:
            result = APIRecovery.call("anthropic", _call)
            return {
                "bull_case": str(result.get("bull_case", "")),
                "confidence_boost": max(-0.2, min(0.2, float(result.get("confidence_boost", 0.0)))),
            }
        except Exception as exc:
            log.warning("BullResearcher failed: %s", exc)
            return {"bull_case": "Analysis unavailable", "confidence_boost": 0.0}


class BearResearcher:
    """Argues AGAINST the trade. Finds risks."""

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def analyze(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        technical_data: dict,
        sentiment_data: dict,
    ) -> dict:
        user_msg = _BEAR_USER.format(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            technical=json.dumps(technical_data, default=str),
            sentiment=json.dumps(sentiment_data, default=str),
        )

        def _call():
            kwargs: dict = {
                "model": CLAUDE_MODEL,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": user_msg}],
            }
            if ENABLE_PROMPT_CACHING:
                kwargs["system"] = [{
                    "type": "text",
                    "text": _BEAR_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                kwargs["system"] = _BEAR_SYSTEM

            msg = self._client.messages.create(**kwargs)
            _log_cache_usage(msg, "BearResearcher")
            return json.loads(_strip_fences(msg.content[0].text))

        try:
            result = APIRecovery.call("anthropic", _call)
            return {
                "bear_case": str(result.get("bear_case", "")),
                "confidence_penalty": max(-0.2, min(0.0, float(result.get("confidence_penalty", 0.0)))),
            }
        except Exception as exc:
            log.warning("BearResearcher failed: %s", exc)
            return {"bear_case": "Analysis unavailable", "confidence_penalty": 0.0}


# ── Debate orchestrator ──────────────────────────────────────────────────────

# Signals that skip debate entirely (no point debating a HOLD)
_SKIP_SIGNALS = {"HOLD", "CONFLICTING"}


class BullBearDebate:
    """Runs both researchers and synthesises a final verdict."""

    def __init__(
        self,
        bull: BullResearcher | None = None,
        bear: BearResearcher | None = None,
    ) -> None:
        self._bull = bull or BullResearcher()
        self._bear = bear or BearResearcher()

    @staticmethod
    def is_enabled() -> bool:
        return os.environ.get("ENABLE_BULL_BEAR_DEBATE", "").lower() in ("true", "1", "yes")

    def run(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        technical_data: dict,
        sentiment_data: dict,
    ) -> DebateResult:
        """Run bull/bear debate and return adjusted signal + confidence."""

        # Skip debate for HOLD / CONFLICTING
        if signal in _SKIP_SIGNALS:
            return DebateResult(
                ticker=ticker,
                original_signal=signal,
                original_confidence=confidence,
                final_signal=signal,
                adjusted_confidence=confidence,
                debate_summary="Debate skipped — signal is {}.".format(signal),
            )

        kwargs = dict(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            technical_data=technical_data,
            sentiment_data=sentiment_data,
        )

        try:
            # Run both researchers (sync — each is a single short API call)
            bull_result = self._bull.analyze(**kwargs)
            bear_result = self._bear.analyze(**kwargs)
        except Exception as exc:
            log.error("BullBearDebate failed: %s — using original signal", exc)
            return DebateResult(
                ticker=ticker,
                original_signal=signal,
                original_confidence=confidence,
                final_signal=signal,
                adjusted_confidence=confidence,
                debate_summary="Debate failed — using original signal.",
                degraded=True,
            )

        return self._synthesise(
            ticker, signal, confidence, bull_result, bear_result,
        )

    async def run_async(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        technical_data: dict,
        sentiment_data: dict,
    ) -> DebateResult:
        """Async version — runs bull and bear in parallel via asyncio."""

        if signal in _SKIP_SIGNALS:
            return DebateResult(
                ticker=ticker,
                original_signal=signal,
                original_confidence=confidence,
                final_signal=signal,
                adjusted_confidence=confidence,
                debate_summary="Debate skipped — signal is {}.".format(signal),
            )

        kwargs = dict(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            technical_data=technical_data,
            sentiment_data=sentiment_data,
        )

        try:
            bull_result, bear_result = await asyncio.gather(
                asyncio.to_thread(self._bull.analyze, **kwargs),
                asyncio.to_thread(self._bear.analyze, **kwargs),
            )
        except Exception as exc:
            log.error("BullBearDebate async failed: %s — using original signal", exc)
            return DebateResult(
                ticker=ticker,
                original_signal=signal,
                original_confidence=confidence,
                final_signal=signal,
                adjusted_confidence=confidence,
                debate_summary="Debate failed — using original signal.",
                degraded=True,
            )

        return self._synthesise(
            ticker, signal, confidence, bull_result, bear_result,
        )

    # ── Private synthesis ────────────────────────────────────────────────

    # Words indicating strong, convincing arguments
    _CONVICTION = frozenset([
        "strong", "compelling", "robust", "significant", "clear",
        "solid", "decisive", "convincing", "healthy",
    ])
    # Words indicating weakness, risk, or problems
    _WEAKNESS = frozenset([
        "weak", "risky", "risk", "vulnerable", "declining",
        "deteriorating", "overvalued", "concern", "fragile",
        "dangerous", "uncertain", "unclear",
    ])
    # Hedging / qualification words that undermine the argument containing them
    _HEDGING = frozenset([
        "however", "but", "despite", "although", "might", "could",
    ])

    @staticmethod
    def _analyze_transcript(bull_case: str, bear_case: str) -> float:
        """Score debate text for bull vs bear strength.

        Positive → bull wins, negative → bear wins.
        """
        bull_words = set(re.findall(r"[a-z]+", bull_case.lower()))
        bear_words = set(re.findall(r"[a-z]+", bear_case.lower()))

        C = BullBearDebate._CONVICTION
        W = BullBearDebate._WEAKNESS
        H = BullBearDebate._HEDGING

        # Bull case: conviction words strengthen, weakness/hedging weaken
        bull_score = (
            len(bull_words & C) * 0.12
            - len(bull_words & W) * 0.10
            - len(bull_words & H) * 0.08
        )

        # Bear case: weakness words = strong bear, hedging = weak bear
        bear_score = (
            len(bear_words & W) * 0.10
            + len(bear_words & C) * 0.08
            - len(bear_words & H) * 0.10
        )

        return bull_score - bear_score

    @staticmethod
    def _synthesise(
        ticker: str,
        signal: str,
        confidence: float,
        bull: dict,
        bear: dict,
    ) -> DebateResult:
        """Merge bull and bear perspectives into a final verdict.

        Returns a dynamic adjustment between -0.15 and +0.10 based on:
            1. Numeric scores from both researchers (60% weight)
            2. Text analysis of argument strength (40% weight)

        Outcome bands:
            Clear bull win   (+0.05 to +0.10)
            Slight bull win  (+0.02 to +0.05)
            Contested        (-0.03 to +0.02)
            Slight bear win  (-0.06 to -0.03) — cautious
            Bear win         (-0.10 to -0.06) — disagree
            Clear bear win   (-0.15 to -0.10) — strong disagree
        """

        bull_boost = bull.get("confidence_boost", 0.0)
        bear_penalty = bear.get("confidence_penalty", 0.0)
        bull_case = bull.get("bull_case", "")
        bear_case = bear.get("bear_case", "")

        # 1. Numeric component: combine both researcher scores
        numeric = bull_boost + bear_penalty

        # 2. Text analysis: who made the stronger argument?
        text_score = BullBearDebate._analyze_transcript(bull_case, bear_case)

        # 3. Weighted combination
        raw = 0.6 * numeric + 0.4 * text_score

        # 4. Clamp to [-0.15, +0.10]
        adjustment = round(max(-0.15, min(0.10, raw)), 4)

        # 5. Classify for summary
        if adjustment > 0.05:
            summary = "Clear bull win — conviction strengthened."
        elif adjustment > 0.02:
            summary = "Slight bull advantage — modest confidence boost."
        elif adjustment > -0.03:
            summary = "Contested debate — signal held with minor adjustment."
        elif adjustment > -0.06:
            summary = "Bear raises cautious concerns — penalty applied."
        elif adjustment > -0.10:
            summary = "Bull and bear disagree — confidence reduced."
        else:
            summary = "Clear bear win — bull and bear disagree significantly."

        adjusted = round(max(0.0, min(1.0, confidence + adjustment)), 2)

        # If confidence drops below 0.15, downgrade signal to HOLD
        final_signal = signal
        if adjusted < 0.15:
            final_signal = "HOLD"
            summary += " Confidence too low — downgraded to HOLD."

        log.debug(
            "Debate %s: boost=%.3f penalty=%.3f text=%.3f adj=%.4f conf=%.2f→%.2f signal=%s→%s",
            ticker, bull_boost, bear_penalty, text_score, adjustment,
            confidence, adjusted, signal, final_signal,
        )

        return DebateResult(
            ticker=ticker,
            original_signal=signal,
            original_confidence=confidence,
            bull_case=bull_case,
            bear_case=bear_case,
            final_signal=final_signal,
            adjusted_confidence=adjusted,
            debate_adjustment=adjustment,
            debate_summary=summary,
        )
