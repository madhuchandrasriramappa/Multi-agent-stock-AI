"""
Phase 5 tests — AI Analyst Agent, prompt builder, and LLM client.

Unit tests are pure — no DB, no real HTTP calls to Azure OpenAI.
The LLMClient automatically uses mock mode when Azure credentials are absent,
so the integration test also works without any cloud configuration.

Run unit tests:        pytest tests/test_analyst.py -v -m "not integration"
Run integration tests: pytest tests/test_analyst.py -v -m integration -s
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_latest(symbol: str = "AAPL") -> dict:
    return {
        "symbol":         symbol,
        "asset_type":     "stock",
        "timestamp":      datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc),
        "close":          175.23,
        "volume":         45_000_000.0,
        "rsi_14":         62.5,
        "macd_histogram": 0.45,
        "volatility_14":  0.0182,
        "vwap":           174.80,
        "sma_20":         172.10,
        "ema_12":         173.50,
    }


def _make_alerts(n: int = 3) -> list[dict]:
    return [
        {
            "timestamp":     datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
            "detector":      "zscore",
            "feature":       "volume",
            "feature_value": 45_000_000.0,
            "score":         3.82,
            "severity":      "low",
        }
    ] * n


# ── Prompt builder ─────────────────────────────────────────────────────────────

def test_build_user_prompt_contains_symbol():
    from agents.analyst.prompt_builder import build_user_prompt
    prompt = build_user_prompt("AAPL", "stock", _make_latest(), [])
    assert "AAPL" in prompt


def test_build_user_prompt_contains_indicators():
    from agents.analyst.prompt_builder import build_user_prompt
    prompt = build_user_prompt("AAPL", "stock", _make_latest(), [])
    assert "RSI" in prompt
    assert "MACD" in prompt
    assert "175.2300" in prompt   # close value formatted to 4dp


def test_build_user_prompt_no_alerts_shows_none():
    from agents.analyst.prompt_builder import build_user_prompt
    prompt = build_user_prompt("BTC", "crypto", _make_latest("BTC"), [])
    assert "None detected" in prompt


def test_build_user_prompt_shows_alert_details():
    from agents.analyst.prompt_builder import build_user_prompt
    alerts = _make_alerts(2)
    prompt = build_user_prompt("AAPL", "stock", _make_latest(), alerts)
    assert "zscore" in prompt
    assert "volume" in prompt
    assert "LOW" in prompt   # severity uppercased


def test_build_user_prompt_caps_alerts_at_10():
    from agents.analyst.prompt_builder import build_user_prompt
    alerts = _make_alerts(15)
    prompt = build_user_prompt("AAPL", "stock", _make_latest(), alerts)
    # Header says 15 total but only 10 shown
    assert "15 recent" in prompt
    assert "showing 10" in prompt


def test_build_user_prompt_handles_none_values():
    """Indicators that are None should render as N/A, not crash."""
    from agents.analyst.prompt_builder import build_user_prompt
    latest = _make_latest()
    latest["rsi_14"] = None
    latest["vwap"] = None
    prompt = build_user_prompt("AAPL", "stock", latest, [])
    assert "N/A" in prompt


def test_build_system_prompt_is_non_empty():
    from agents.analyst.prompt_builder import build_system_prompt
    p = build_system_prompt()
    assert len(p) > 50
    assert "OUTLOOK" in p


def test_extract_outlook_bullish():
    from agents.analyst.prompt_builder import extract_outlook
    assert extract_outlook("Strong momentum. OUTLOOK: BULLISH") == "BULLISH"


def test_extract_outlook_bearish():
    from agents.analyst.prompt_builder import extract_outlook
    assert extract_outlook("Risk elevated. OUTLOOK: BEARISH.") == "BEARISH"


def test_extract_outlook_neutral():
    from agents.analyst.prompt_builder import extract_outlook
    assert extract_outlook("Mixed signals. OUTLOOK: NEUTRAL") == "NEUTRAL"


def test_extract_outlook_unknown():
    from agents.analyst.prompt_builder import extract_outlook
    assert extract_outlook("No outlook present in this text.") == "UNKNOWN"


# ── LLM client ─────────────────────────────────────────────────────────────────

def test_llm_client_mock_when_no_credentials():
    """Without Azure credentials (default in dev), LLMClient must use mock mode."""
    from agents.analyst.llm_client import LLMClient
    with patch("agents.analyst.llm_client.settings") as mock_settings:
        mock_settings.azure_openai_endpoint = None
        mock_settings.azure_openai_api_key  = None
        client = LLMClient()
    assert client.is_mock


def test_llm_client_mock_returns_neutral_report():
    from agents.analyst.llm_client import LLMClient
    with patch("agents.analyst.llm_client.settings") as mock_settings:
        mock_settings.azure_openai_endpoint = None
        mock_settings.azure_openai_api_key  = None
        client = LLMClient()
    text, pt, ct = client.complete("sys", "user")
    assert isinstance(text, str)
    assert len(text) > 20
    assert pt == 0 and ct == 0
    assert "OUTLOOK: NEUTRAL" in text


# ── AnalystAgent (mocked DB + LLM) ────────────────────────────────────────────

def _mock_llm():
    m = MagicMock()
    m.is_mock = True
    m.complete.return_value = (
        "Indicators show balanced conditions. Volume anomaly noted. "
        "RSI at 62.5 is approaching overbought territory. "
        "OUTLOOK: NEUTRAL",
        120,
        55,
    )
    return m


@patch("agents.analyst.analyst_agent.AnalystAgent._save_report")
@patch("agents.analyst.analyst_agent.AnalystAgent._load_recent_alerts")
@patch("agents.analyst.analyst_agent.AnalystAgent._load_latest_features")
@patch("agents.analyst.analyst_agent.AnalystAgent._get_symbols")
@patch("agents.analyst.analyst_agent.LLMClient")
def test_analyst_agent_execute_success(
    MockLLM, mock_symbols, mock_features, mock_alerts, mock_save
):
    from agents.analyst.analyst_agent import AnalystAgent

    MockLLM.return_value = _mock_llm()
    mock_symbols.return_value  = ["AAPL"]
    mock_features.return_value = _make_latest()
    mock_alerts.return_value   = _make_alerts(2)

    result = AnalystAgent().execute()

    assert result.succeeded
    assert result.data["reports_generated"] == 1
    assert "AAPL" in result.data["symbols_analyzed"]
    assert result.data["mode"] == "mock"
    assert result.data["total_prompt_tokens"] == 120
    assert result.data["total_completion_tokens"] == 55
    mock_save.assert_called_once()


@patch("agents.analyst.analyst_agent.AnalystAgent._save_report")
@patch("agents.analyst.analyst_agent.AnalystAgent._load_recent_alerts")
@patch("agents.analyst.analyst_agent.AnalystAgent._load_latest_features")
@patch("agents.analyst.analyst_agent.AnalystAgent._get_symbols")
@patch("agents.analyst.analyst_agent.LLMClient")
def test_analyst_agent_multi_symbol(
    MockLLM, mock_symbols, mock_features, mock_alerts, mock_save
):
    from agents.analyst.analyst_agent import AnalystAgent

    MockLLM.return_value = _mock_llm()
    mock_symbols.return_value  = ["AAPL", "bitcoin"]
    mock_features.return_value = _make_latest()
    mock_alerts.return_value   = []

    result = AnalystAgent().execute()

    assert result.succeeded
    assert result.data["reports_generated"] == 2
    assert mock_save.call_count == 2


@patch("agents.analyst.analyst_agent.AnalystAgent._get_symbols", return_value=[])
@patch("agents.analyst.analyst_agent.LLMClient")
def test_analyst_agent_no_symbols(MockLLM, mock_symbols):
    from agents.analyst.analyst_agent import AnalystAgent

    result = AnalystAgent().execute()

    assert result.succeeded
    assert result.data["reports_generated"] == 0
    assert result.data["symbols_analyzed"] == []
    MockLLM.return_value.complete.assert_not_called()


@patch("agents.analyst.analyst_agent.AnalystAgent._save_report")
@patch("agents.analyst.analyst_agent.AnalystAgent._load_recent_alerts")
@patch("agents.analyst.analyst_agent.AnalystAgent._load_latest_features", return_value=None)
@patch("agents.analyst.analyst_agent.AnalystAgent._get_symbols", return_value=["AAPL"])
@patch("agents.analyst.analyst_agent.LLMClient")
def test_analyst_agent_skips_symbol_without_features(
    MockLLM, mock_symbols, mock_features, mock_alerts, mock_save
):
    from agents.analyst.analyst_agent import AnalystAgent

    MockLLM.return_value = _mock_llm()
    result = AnalystAgent().execute()

    assert result.succeeded
    assert result.data["reports_generated"] == 0
    mock_save.assert_not_called()


# ── Integration test ───────────────────────────────────────────────────────────

@pytest.mark.integration
def test_full_pipeline_with_analyst():
    """
    Full pipeline: Ingestion → Cleaning → Features → Anomalies → Analysis.
    The analyst always runs in mock mode here (no Azure credentials in local dev).
    """
    from agents.analyst.analyst_agent import AnalystAgent
    from agents.anomaly_detection.anomaly_agent import AnomalyAgent
    from agents.cleaning.cleaning_agent import CleaningAgent
    from agents.feature_engineering.feature_agent import FeatureAgent
    from agents.ingestion.ingestion_agent import IngestionAgent
    from db.migrate import run_migrations

    run_migrations()

    ingest = IngestionAgent().execute(payload={
        "stock_symbols":  ["AAPL"],
        "crypto_symbols": ["bitcoin"],
        "period": "5d",
        "days":   7,
    })
    assert ingest.succeeded

    all_records = ingest.data["stocks"] + ingest.data["crypto"]
    clean = CleaningAgent().execute(payload={"records": all_records})
    assert clean.succeeded

    features = FeatureAgent().execute()
    assert features.succeeded

    anomalies = AnomalyAgent().execute()
    assert anomalies.succeeded

    analysis = AnalystAgent().execute()
    assert analysis.succeeded
    assert analysis.data["reports_generated"] > 0

    print(f"\n  Reports generated  : {analysis.data['reports_generated']}")
    print(f"  Symbols analyzed   : {analysis.data['symbols_analyzed']}")
    print(f"  Mode               : {analysis.data['mode']}")
    print(f"  Prompt tokens      : {analysis.data['total_prompt_tokens']}")
    print(f"  Completion tokens  : {analysis.data['total_completion_tokens']}")
