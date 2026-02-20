"""
Unit tests for SentimentAgent.

All tests run fully offline — the Anthropic client is mocked via the
``mock_anthropic_client`` fixture defined in conftest.py.

Test classes
------------
TestClaudeClassify      Happy-path Claude API call and result structure.
TestFallbackBehavior    Fallback to rule-based scorer on various failure modes.
TestResultStructure     Output shape invariants for both code paths.
TestConfidenceScores    Confidence constants for Claude vs. rule-based results.

Integration tests (require real ANTHROPIC_API_KEY):
TestLiveClaudeCall      @pytest.mark.integration — real API round-trip.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("NEWSAPI_KEY",       "test-key-not-real")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_claude_response(sentiment: str = "bullish", reason: str = "test") -> MagicMock:
    """Build a mock Anthropic message response."""
    payload = json.dumps({"sentiment": sentiment, "reason": reason})
    msg = MagicMock()
    msg.content = [MagicMock(text=payload)]
    return msg


def _make_agent():
    from agents.sentiment_agent import SentimentAgent
    return SentimentAgent(api_key="test-key")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Claude classify — happy path
# ══════════════════════════════════════════════════════════════════════════════

class TestClaudeClassify:
    """Claude API is mocked; verifies the result dict is shaped correctly."""

    def test_bullish_result(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = _make_claude_response("bullish")
        agent = _make_agent()
        result = agent.run("Apple hits all-time high on record earnings", "AAPL")

        assert result["sentiment"] == "bullish"
        assert result["score"]     == 1
        assert result["degraded"]  is False
        assert result["confidence"] == pytest.approx(0.85)
        assert "reason" in result
        assert result["headline"] == "Apple hits all-time high on record earnings"

    def test_bearish_result(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = _make_claude_response("bearish", "Poor outlook")
        agent = _make_agent()
        result = agent.run("Tesla misses earnings by wide margin", "TSLA")

        assert result["sentiment"] == "bearish"
        assert result["score"]     == -1
        assert result["degraded"]  is False

    def test_neutral_result(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = _make_claude_response("neutral", "Mixed signals")
        agent = _make_agent()
        result = agent.run("Apple holds annual shareholder meeting", "AAPL")

        assert result["sentiment"] == "neutral"
        assert result["score"]     == 0

    def test_claude_is_called_once_per_headline(self, mock_anthropic_client):
        agent = _make_agent()
        agent.run("headline one", "AAPL")
        agent.run("headline two", "AAPL")
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_ticker_appears_in_prompt(self, mock_anthropic_client):
        agent = _make_agent()
        agent.run("some headline", "NVDA")
        call_kwargs = mock_anthropic_client.messages.create.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][2]
        prompt_text = str(messages)
        assert "NVDA" in prompt_text

    def test_headline_appears_in_result(self, mock_anthropic_client):
        headline = "Unique headline text for testing"
        agent = _make_agent()
        result = agent.run(headline, "AAPL")
        assert result["headline"] == headline


# ══════════════════════════════════════════════════════════════════════════════
# 2. Fallback behaviour
# ══════════════════════════════════════════════════════════════════════════════

class TestFallbackBehavior:
    """When the Anthropic API fails the agent must fall back to rule-based scoring."""

    def _run_with_api_error(self, exception, headline="Apple surges on earnings"):
        with patch("agents.sentiment_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = exception
            mock_cls.return_value = mock_client
            with patch("utils.api_recovery.time.sleep"):  # skip retry waits
                agent = _make_agent()
                return agent.run(headline, "AAPL")

    def test_fallback_on_generic_exception(self):
        result = self._run_with_api_error(RuntimeError("connection refused"))
        assert result["degraded"]   is True
        assert result["confidence"] == pytest.approx(0.55)
        assert result["sentiment"]  in ("bullish", "bearish", "neutral")

    def test_fallback_on_json_decode_error(self):
        # Claude returns malformed JSON → JSONDecodeError in _claude_classify
        with patch("agents.sentiment_agent.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            msg = MagicMock()
            msg.content = [MagicMock(text="not json")]
            mock_client.messages.create.return_value = msg
            mock_cls.return_value = mock_client
            with patch("utils.api_recovery.time.sleep"):
                agent = _make_agent()
                result = agent.run("Apple surges to record", "AAPL")
        assert result["degraded"] is True

    def test_fallback_degraded_flag_is_true(self):
        result = self._run_with_api_error(ConnectionError("timeout"))
        assert result["degraded"] is True

    def test_fallback_result_has_headline(self):
        headline = "Company announces record profit"
        result = self._run_with_api_error(RuntimeError("api down"), headline)
        assert result["headline"] == headline

    def test_fallback_score_is_valid_int(self):
        result = self._run_with_api_error(RuntimeError("fail"))
        assert result["score"] in (-1, 0, 1)

    def test_bullish_headline_scores_positive_in_fallback(self):
        result = self._run_with_api_error(
            RuntimeError("api down"),
            headline="Apple beats expectations with record earnings surge"
        )
        # Strong bullish keywords → should land bullish or at minimum neutral
        assert result["sentiment"] in ("bullish", "neutral")

    def test_bearish_headline_scores_negative_in_fallback(self):
        result = self._run_with_api_error(
            RuntimeError("api down"),
            headline="Company faces fraud investigation and bankruptcy filing"
        )
        assert result["sentiment"] == "bearish"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Result structure invariants
# ══════════════════════════════════════════════════════════════════════════════

class TestResultStructure:
    """Both the Claude path and the fallback path must return the same keys."""

    REQUIRED_KEYS = {"sentiment", "score", "reason", "headline", "degraded", "confidence"}

    def test_claude_path_has_all_keys(self, mock_anthropic_client):
        agent = _make_agent()
        result = agent.run("Apple earnings beat", "AAPL")
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_fallback_path_has_all_keys(self):
        with patch("agents.sentiment_agent.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = RuntimeError("fail")
            with patch("utils.api_recovery.time.sleep"):
                agent = _make_agent()
                result = agent.run("Tesla drops on weak sales", "TSLA")
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_sentiment_is_valid_string(self, mock_anthropic_client):
        agent = _make_agent()
        result = agent.run("headline", "AAPL")
        assert result["sentiment"] in ("bullish", "bearish", "neutral")

    def test_score_matches_sentiment(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = _make_claude_response("bearish")
        agent = _make_agent()
        result = agent.run("headline", "AAPL")
        assert result["score"] == -1

    def test_degraded_is_bool(self, mock_anthropic_client):
        agent = _make_agent()
        result = agent.run("headline", "AAPL")
        assert isinstance(result["degraded"], bool)

    def test_confidence_is_float(self, mock_anthropic_client):
        agent = _make_agent()
        result = agent.run("headline", "AAPL")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. Confidence scores
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidenceScores:

    def test_claude_confidence_is_0_85(self, mock_anthropic_client):
        agent = _make_agent()
        result = agent.run("Apple surges", "AAPL")
        assert result["confidence"] == pytest.approx(0.85)

    def test_fallback_confidence_is_0_55(self):
        with patch("agents.sentiment_agent.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = RuntimeError("fail")
            with patch("utils.api_recovery.time.sleep"):
                agent = _make_agent()
                result = agent.run("Apple surges", "AAPL")
        assert result["confidence"] == pytest.approx(0.55)

    def test_claude_not_degraded(self, mock_anthropic_client):
        agent = _make_agent()
        result = agent.run("headline", "AAPL")
        assert result["degraded"] is False

    def test_fallback_is_degraded(self):
        with patch("agents.sentiment_agent.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = RuntimeError("fail")
            with patch("utils.api_recovery.time.sleep"):
                agent = _make_agent()
                result = agent.run("headline", "AAPL")
        assert result["degraded"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 5. Integration tests (real API — skipped by default)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestLiveClaudeCall:
    """
    Requires a real ANTHROPIC_API_KEY in the environment.

    Run with:
        pytest -m integration tests/test_sentiment_agent.py
    """

    def test_live_bullish_headline(self):
        import os
        if not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-"):
            pytest.skip("ANTHROPIC_API_KEY not set or is a dummy key")
        from agents.sentiment_agent import SentimentAgent
        agent = SentimentAgent()
        result = agent.run("Apple reports record earnings, stock surges 5%", "AAPL")
        assert result["sentiment"] in ("bullish", "neutral", "bearish")
        assert result["degraded"] is False
        assert result["confidence"] == pytest.approx(0.85)

    def test_live_bearish_headline(self):
        import os
        if not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-"):
            pytest.skip("ANTHROPIC_API_KEY not set or is a dummy key")
        from agents.sentiment_agent import SentimentAgent
        agent = SentimentAgent()
        result = agent.run("Tesla faces SEC investigation and misses earnings badly", "TSLA")
        assert result["sentiment"] in ("bullish", "neutral", "bearish")
        assert result["degraded"] is False
