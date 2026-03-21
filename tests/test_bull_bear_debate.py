"""Tests for the bull/bear debate agent."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bull_bear_debate import (
    BearResearcher,
    BullBearDebate,
    BullResearcher,
    DebateResult,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_claude_response(payload: dict) -> MagicMock:
    """Build a mock Anthropic message with the given JSON payload."""
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(payload))]
    return msg


def _mock_client(payload: dict) -> MagicMock:
    """Build a mock anthropic.Anthropic client returning *payload*."""
    client = MagicMock()
    client.messages.create.return_value = _mock_claude_response(payload)
    return client


_TECH_DATA = {"rsi": 55.0, "price": 150.0, "macd_histogram": 0.5}
_SENT_DATA = {"signal": "BUY", "avg_score": 0.3}


# ── BullResearcher ───────────────────────────────────────────────────────────

class TestBullResearcher:

    def test_returns_bull_case_and_boost(self):
        client = _mock_client({"bull_case": "Strong momentum.", "confidence_boost": 0.1})
        bull = BullResearcher(client=client)
        result = bull.analyze("AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA)
        assert result["bull_case"] == "Strong momentum."
        assert result["confidence_boost"] == 0.1

    def test_clamps_boost_to_max(self):
        client = _mock_client({"bull_case": "x", "confidence_boost": 0.99})
        result = BullResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["confidence_boost"] == 0.2

    def test_clamps_boost_to_min(self):
        client = _mock_client({"bull_case": "x", "confidence_boost": -0.99})
        result = BullResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["confidence_boost"] == -0.2

    def test_fallback_on_api_error(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        result = BullResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["bull_case"] == "Analysis unavailable"
        assert result["confidence_boost"] == 0.0


# ── BearResearcher ───────────────────────────────────────────────────────────

class TestBearResearcher:

    def test_returns_bear_case_and_penalty(self):
        client = _mock_client({"bear_case": "RSI divergence.", "confidence_penalty": -0.1})
        bear = BearResearcher(client=client)
        result = bear.analyze("AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA)
        assert result["bear_case"] == "RSI divergence."
        assert result["confidence_penalty"] == -0.1

    def test_clamps_penalty_to_zero(self):
        client = _mock_client({"bear_case": "x", "confidence_penalty": 0.5})
        result = BearResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["confidence_penalty"] == 0.0

    def test_clamps_penalty_to_min(self):
        client = _mock_client({"bear_case": "x", "confidence_penalty": -0.99})
        result = BearResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["confidence_penalty"] == -0.2

    def test_fallback_on_api_error(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        result = BearResearcher(client=client).analyze("AAPL", "BUY", 0.5, {}, {})
        assert result["bear_case"] == "Analysis unavailable"
        assert result["confidence_penalty"] == 0.0


# ── BullBearDebate ───────────────────────────────────────────────────────────

class TestBullBearDebate:

    def _make_debate(self, bull_payload: dict, bear_payload: dict) -> BullBearDebate:
        bull = BullResearcher(client=_mock_client(bull_payload))
        bear = BearResearcher(client=_mock_client(bear_payload))
        return BullBearDebate(bull=bull, bear=bear)

    # -- HOLD signals skip debate --

    def test_hold_skips_debate(self):
        debate = self._make_debate({}, {})
        result = debate.run("AAPL", "HOLD", 0.25, {}, {})
        assert result.final_signal == "HOLD"
        assert result.adjusted_confidence == 0.25
        assert "skipped" in result.debate_summary.lower()

    def test_conflicting_skips_debate(self):
        debate = self._make_debate({}, {})
        result = debate.run("AAPL", "CONFLICTING", 0.10, {}, {})
        assert result.final_signal == "CONFLICTING"
        assert result.adjusted_confidence == 0.10

    # -- Agreement boosts confidence --

    def test_agreement_boosts_confidence(self):
        debate = self._make_debate(
            {"bull_case": "Strong trend.", "confidence_boost": 0.15},
            {"bear_case": "Minor risk.", "confidence_penalty": -0.02},
        )
        result = debate.run("AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA)
        assert result.adjusted_confidence > result.original_confidence
        assert result.final_signal == "STRONG BUY"
        assert "agree" in result.debate_summary.lower()

    # -- Disagreement reduces confidence --

    def test_disagreement_reduces_confidence(self):
        debate = self._make_debate(
            {"bull_case": "Uncertain.", "confidence_boost": -0.05},
            {"bear_case": "Major risk.", "confidence_penalty": -0.15},
        )
        result = debate.run("AAPL", "WEAK BUY", 0.45, _TECH_DATA, _SENT_DATA)
        assert result.adjusted_confidence < result.original_confidence
        assert "doubt" in result.debate_summary.lower()

    # -- Mixed perspectives --

    def test_mixed_applies_cautious_adjustment(self):
        debate = self._make_debate(
            {"bull_case": "Upside potential.", "confidence_boost": 0.1},
            {"bear_case": "Earnings risk.", "confidence_penalty": -0.15},
        )
        result = debate.run("AAPL", "WEAK BUY", 0.45, _TECH_DATA, _SENT_DATA)
        assert result.final_signal == "WEAK BUY"
        assert "disagree" in result.debate_summary.lower()

    # -- Very low confidence downgrades to HOLD --

    def test_low_confidence_downgrades_to_hold(self):
        debate = self._make_debate(
            {"bull_case": "Weak.", "confidence_boost": -0.1},
            {"bear_case": "Very risky.", "confidence_penalty": -0.2},
        )
        result = debate.run("AAPL", "WEAK BUY", 0.20, _TECH_DATA, _SENT_DATA)
        assert result.final_signal == "HOLD"
        assert "downgraded" in result.debate_summary.lower()

    # -- Graceful fallback on debate failure --

    def test_fallback_on_debate_failure(self):
        bull = MagicMock()
        bull.analyze.side_effect = RuntimeError("catastrophic")
        bear = BearResearcher(client=_mock_client({"bear_case": "x", "confidence_penalty": -0.1}))
        debate = BullBearDebate(bull=bull, bear=bear)
        result = debate.run("AAPL", "STRONG BUY", 0.70, {}, {})
        assert result.final_signal == "STRONG BUY"
        assert result.adjusted_confidence == 0.70
        assert result.degraded is True

    # -- DebateResult dataclass --

    def test_debate_result_fields(self):
        r = DebateResult(
            ticker="AAPL",
            original_signal="STRONG BUY",
            original_confidence=0.65,
            bull_case="Good momentum",
            bear_case="RSI overbought",
            final_signal="STRONG BUY",
            adjusted_confidence=0.72,
            debate_summary="Bull and bear broadly agree.",
        )
        assert r.ticker == "AAPL"
        assert r.degraded is False

    # -- Confidence clamped to [0, 1] --

    def test_confidence_clamped_high(self):
        debate = self._make_debate(
            {"bull_case": "x", "confidence_boost": 0.2},
            {"bear_case": "x", "confidence_penalty": 0.0},
        )
        result = debate.run("AAPL", "STRONG BUY", 0.95, {}, {})
        assert result.adjusted_confidence <= 1.0

    def test_confidence_clamped_low(self):
        debate = self._make_debate(
            {"bull_case": "x", "confidence_boost": -0.2},
            {"bear_case": "x", "confidence_penalty": -0.2},
        )
        result = debate.run("AAPL", "WEAK BUY", 0.05, {}, {})
        assert result.adjusted_confidence >= 0.0


# ── Async path ───────────────────────────────────────────────────────────────

class TestBullBearDebateAsync:

    def _make_debate(self, bull_payload: dict, bear_payload: dict) -> BullBearDebate:
        bull = BullResearcher(client=_mock_client(bull_payload))
        bear = BearResearcher(client=_mock_client(bear_payload))
        return BullBearDebate(bull=bull, bear=bear)

    def test_async_agreement(self):
        debate = self._make_debate(
            {"bull_case": "Strong.", "confidence_boost": 0.1},
            {"bear_case": "Minor.", "confidence_penalty": -0.02},
        )
        result = asyncio.run(debate.run_async(
            "AAPL", "STRONG BUY", 0.65, _TECH_DATA, _SENT_DATA,
        ))
        assert result.adjusted_confidence > result.original_confidence

    def test_async_hold_skips(self):
        debate = self._make_debate({}, {})
        result = asyncio.run(debate.run_async("AAPL", "HOLD", 0.25, {}, {}))
        assert result.final_signal == "HOLD"

    def test_async_fallback_on_failure(self):
        bull = MagicMock()
        bull.analyze.side_effect = RuntimeError("fail")
        bear = MagicMock()
        bear.analyze.side_effect = RuntimeError("fail")
        debate = BullBearDebate(bull=bull, bear=bear)
        result = asyncio.run(debate.run_async("AAPL", "STRONG BUY", 0.7, {}, {}))
        assert result.final_signal == "STRONG BUY"
        assert result.degraded is True


# ── is_enabled flag ──────────────────────────────────────────────────────────

class TestIsEnabled:

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENABLE_BULL_BEAR_DEBATE", None)
            assert BullBearDebate.is_enabled() is False

    def test_enabled_with_true(self):
        with patch.dict(os.environ, {"ENABLE_BULL_BEAR_DEBATE": "true"}):
            assert BullBearDebate.is_enabled() is True

    def test_enabled_with_one(self):
        with patch.dict(os.environ, {"ENABLE_BULL_BEAR_DEBATE": "1"}):
            assert BullBearDebate.is_enabled() is True

    def test_disabled_with_false(self):
        with patch.dict(os.environ, {"ENABLE_BULL_BEAR_DEBATE": "false"}):
            assert BullBearDebate.is_enabled() is False
