"""
Phase 1 tests — Ingestion Agent, Yahoo Finance client, CoinGecko client.

Unit tests use mocks and run instantly with no network access.
Integration tests hit real APIs and are marked @pytest.mark.integration.

Run unit tests only:   pytest tests/test_ingestion.py -v
Run all incl. live:    pytest tests/test_ingestion.py -v -m integration
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Retry decorator ────────────────────────────────────────────────────────────


def test_retry_succeeds_first_try():
    from agents.ingestion.retry import with_retry

    calls = []

    @with_retry(max_attempts=3, base_delay=0)
    def succeed():
        calls.append(1)
        return "ok"

    assert succeed() == "ok"
    assert len(calls) == 1


def test_retry_retries_on_transient_error():
    from agents.ingestion.retry import with_retry

    calls = []

    @with_retry(max_attempts=3, base_delay=0)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return "recovered"

    assert flaky() == "recovered"
    assert len(calls) == 3


def test_retry_raises_after_all_attempts_exhausted():
    from agents.ingestion.retry import with_retry

    @with_retry(max_attempts=2, base_delay=0)
    def always_fails():
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        always_fails()


def test_retry_reraises_immediately_for_excluded_types():
    from agents.ingestion.retry import with_retry

    calls = []

    @with_retry(max_attempts=3, base_delay=0, reraise_on=(ValueError,))
    def bad_input():
        calls.append(1)
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        bad_input()

    # Must not retry — ValueError should propagate on first attempt
    assert len(calls) == 1


# ── Rate limiter ───────────────────────────────────────────────────────────────


def test_rate_limiter_enforces_minimum_interval():
    from agents.ingestion.rate_limiter import RateLimiter

    limiter = RateLimiter(calls_per_minute=120)  # 0.5 s interval
    t0 = time.monotonic()
    limiter.acquire()
    limiter.acquire()
    elapsed = time.monotonic() - t0
    # Should have slept at least 0.4 s (allow some slack for slow CI)
    assert elapsed >= 0.4


def test_rate_limiter_rejects_zero_rpm():
    from agents.ingestion.rate_limiter import RateLimiter

    with pytest.raises(ValueError):
        RateLimiter(calls_per_minute=0)


# ── MarketRecord schema ────────────────────────────────────────────────────────


def test_market_record_to_dict_keys():
    from agents.ingestion.schemas import MarketRecord

    now = datetime.now(timezone.utc)
    record = MarketRecord(
        symbol="AAPL",
        asset_type="stock",
        timestamp=now,
        open=182.0,
        high=185.0,
        low=181.0,
        close=184.0,
        volume=50_000_000,
        source="yahoo_finance",
        fetched_at=now,
        interval="1h",
    )
    d = record.to_dict()
    expected_keys = {
        "symbol", "asset_type", "timestamp", "open", "high",
        "low", "close", "volume", "source", "fetched_at", "interval",
    }
    assert set(d.keys()) == expected_keys
    assert d["symbol"] == "AAPL"
    assert d["asset_type"] == "stock"


# ── Yahoo Finance client ───────────────────────────────────────────────────────


def _make_fake_yf_df() -> pd.DataFrame:
    """Build a minimal DataFrame that mimics yfinance output."""
    index = pd.to_datetime(
        ["2024-01-15 14:00:00", "2024-01-15 15:00:00", "2024-01-15 16:00:00"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "Open":   [182.0, 183.5, 184.0],
            "High":   [185.0, 185.5, 185.0],
            "Low":    [181.0, 182.5, 183.0],
            "Close":  [183.5, 184.0, 184.5],
            "Volume": [10_000_000, 8_000_000, 7_000_000],
        },
        index=index,
    )


@patch("yfinance.Ticker")
def test_yahoo_client_parses_dataframe(mock_ticker_cls):
    from agents.ingestion.yahoo_client import YahooFinanceClient

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _make_fake_yf_df()
    mock_ticker_cls.return_value = mock_ticker

    client = YahooFinanceClient()
    records = client.fetch_symbol("AAPL")

    assert len(records) == 3
    assert records[0].symbol == "AAPL"
    assert records[0].asset_type == "stock"
    assert records[0].source == "yahoo_finance"
    assert records[0].open == 182.0
    assert records[0].volume == 10_000_000
    # Timestamps must be UTC-aware
    assert records[0].timestamp.tzinfo is not None


@patch("yfinance.Ticker")
def test_yahoo_client_returns_empty_on_no_data(mock_ticker_cls):
    from agents.ingestion.yahoo_client import YahooFinanceClient

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker_cls.return_value = mock_ticker

    client = YahooFinanceClient()
    records = client.fetch_symbol("INVALID_TICKER_XYZ")
    assert records == []


@patch("yfinance.Ticker")
def test_yahoo_client_drops_nan_rows(mock_ticker_cls):
    from agents.ingestion.yahoo_client import YahooFinanceClient

    df = _make_fake_yf_df()
    df.loc[df.index[1], "Close"] = float("nan")  # corrupt the middle row

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = df
    mock_ticker_cls.return_value = mock_ticker

    client = YahooFinanceClient()
    records = client.fetch_symbol("AAPL")
    assert len(records) == 2  # NaN row dropped


@patch("yfinance.Ticker")
def test_yahoo_fetch_symbols_collects_errors(mock_ticker_cls):
    from agents.ingestion.yahoo_client import YahooFinanceClient

    call_count = 0

    def side_effect(sym):
        nonlocal call_count
        call_count += 1
        m = MagicMock()
        if sym == "BAD":
            m.history.side_effect = ConnectionError("timeout")
        else:
            m.history.return_value = _make_fake_yf_df()
        return m

    mock_ticker_cls.side_effect = side_effect

    client = YahooFinanceClient()
    records, errors = client.fetch_symbols(["AAPL", "BAD", "MSFT"])

    # AAPL and MSFT should succeed; BAD should be in errors
    assert len(errors) == 1
    assert "BAD" in errors
    assert len(records) == 6  # 3 rows × 2 good symbols


# ── CoinGecko client ───────────────────────────────────────────────────────────


def _mock_ohlc_response() -> list[list]:
    base_ts = 1705276800000  # 2024-01-15 00:00 UTC in ms
    candles = []
    for i in range(6):  # 6 × 4-hour candles
        ts = base_ts + i * 4 * 3600 * 1000
        candles.append([ts, 42000.0 + i, 43000.0, 41500.0, 42500.0 + i])
    return candles


def _mock_market_chart_response() -> dict:
    base_ts = 1705276800000
    volumes = [[base_ts + i * 3600 * 1000, 5e8] for i in range(24)]
    return {"prices": [], "market_caps": [], "total_volumes": volumes}


@patch("requests.Session.get")
def test_coingecko_client_parses_response(mock_get):
    from agents.ingestion.coingecko_client import CoinGeckoClient

    ohlc_resp = MagicMock()
    ohlc_resp.json.return_value = _mock_ohlc_response()
    ohlc_resp.raise_for_status = MagicMock()

    chart_resp = MagicMock()
    chart_resp.json.return_value = _mock_market_chart_response()
    chart_resp.raise_for_status = MagicMock()

    # First call → OHLC, second call → market chart
    mock_get.side_effect = [ohlc_resp, chart_resp]

    client = CoinGeckoClient()
    # Bypass rate limiter in unit tests
    client._rate_limiter.acquire = MagicMock()

    records = client.fetch_coin("bitcoin")

    assert len(records) == 6
    assert records[0].symbol == "bitcoin"
    assert records[0].asset_type == "crypto"
    assert records[0].source == "coingecko"
    assert records[0].interval == "4h"
    assert records[0].open == 42000.0
    assert records[0].timestamp.tzinfo is not None


@patch("requests.Session.get")
def test_coingecko_client_handles_http_error(mock_get):
    import requests as req

    from agents.ingestion.coingecko_client import CoinGeckoClient

    resp = MagicMock()
    resp.raise_for_status.side_effect = req.HTTPError("404 Not Found")
    mock_get.return_value = resp

    client = CoinGeckoClient()
    client._rate_limiter.acquire = MagicMock()

    with pytest.raises(req.HTTPError):
        client.fetch_coin("invalid-coin-xyz")


@patch("requests.Session.get")
def test_coingecko_fetch_coins_collects_errors(mock_get):
    import requests as req

    from agents.ingestion.coingecko_client import CoinGeckoClient

    good_ohlc = MagicMock()
    good_ohlc.json.return_value = _mock_ohlc_response()
    good_ohlc.raise_for_status = MagicMock()

    good_chart = MagicMock()
    good_chart.json.return_value = _mock_market_chart_response()
    good_chart.raise_for_status = MagicMock()

    bad_resp = MagicMock()
    bad_resp.raise_for_status.side_effect = req.HTTPError("404")

    # bitcoin: 2 good calls, bad-coin: 1 bad call (ohlc fails immediately)
    mock_get.side_effect = [good_ohlc, good_chart, bad_resp]

    client = CoinGeckoClient()
    client._rate_limiter.acquire = MagicMock()

    records, errors = client.fetch_coins(["bitcoin", "bad-coin"])

    assert len(errors) == 1
    assert "bad-coin" in errors
    assert len(records) == 6


# ── Ingestion Agent ────────────────────────────────────────────────────────────


@patch.object(
    __import__("agents.ingestion.coingecko_client", fromlist=["CoinGeckoClient"]).CoinGeckoClient,
    "fetch_coins",
)
@patch.object(
    __import__("agents.ingestion.yahoo_client", fromlist=["YahooFinanceClient"]).YahooFinanceClient,
    "fetch_symbols",
)
def test_ingestion_agent_returns_agent_result(mock_yahoo, mock_cg):
    from agents.ingestion.schemas import MarketRecord
    from agents.ingestion.ingestion_agent import IngestionAgent

    now = datetime.now(timezone.utc)
    fake_stock = MarketRecord("AAPL", "stock", now, 182.0, 185.0, 181.0, 184.0, 5e7, "yahoo_finance", now, "1h")
    fake_crypto = MarketRecord("bitcoin", "crypto", now, 42000.0, 43000.0, 41500.0, 42500.0, 1e9, "coingecko", now, "4h")

    mock_yahoo.return_value = ([fake_stock], {})
    mock_cg.return_value = ([fake_crypto], {})

    agent = IngestionAgent()
    result = agent.execute()

    assert result.succeeded
    assert result.agent == "ingestion_agent"
    assert result.data["summary"]["total_records"] == 2
    assert result.data["summary"]["stock_records"] == 1
    assert result.data["summary"]["crypto_records"] == 1
    assert result.data["errors"]["stocks"] == {}
    assert result.data["errors"]["crypto"] == {}


@patch.object(
    __import__("agents.ingestion.coingecko_client", fromlist=["CoinGeckoClient"]).CoinGeckoClient,
    "fetch_coins",
)
@patch.object(
    __import__("agents.ingestion.yahoo_client", fromlist=["YahooFinanceClient"]).YahooFinanceClient,
    "fetch_symbols",
)
def test_ingestion_agent_handles_partial_failures(mock_yahoo, mock_cg):
    from agents.ingestion.schemas import MarketRecord
    from agents.ingestion.ingestion_agent import IngestionAgent

    now = datetime.now(timezone.utc)
    fake_stock = MarketRecord("AAPL", "stock", now, 182.0, 185.0, 181.0, 184.0, 5e7, "yahoo_finance", now, "1h")

    mock_yahoo.return_value = ([fake_stock], {"MSFT": "timeout"})
    mock_cg.return_value = ([], {"bitcoin": "429 rate limited"})

    agent = IngestionAgent()
    result = agent.execute()

    # Agent-level result should still succeed (partial data is valid)
    assert result.succeeded
    assert result.data["summary"]["stock_records"] == 1
    assert result.data["summary"]["crypto_records"] == 0
    assert "MSFT" in result.data["errors"]["stocks"]
    assert "bitcoin" in result.data["errors"]["crypto"]


# ── Integration tests (skipped unless -m integration flag is passed) ───────────


@pytest.mark.integration
def test_yahoo_live_fetch_aapl():
    from agents.ingestion.yahoo_client import YahooFinanceClient

    client = YahooFinanceClient(period="2d", interval="1h")
    records = client.fetch_symbol("AAPL")
    assert len(records) > 0
    assert all(r.symbol == "AAPL" for r in records)
    assert all(r.close > 0 for r in records)


@pytest.mark.integration
def test_coingecko_live_fetch_bitcoin():
    from agents.ingestion.coingecko_client import CoinGeckoClient

    client = CoinGeckoClient(days=1)
    records = client.fetch_coin("bitcoin")
    assert len(records) > 0
    assert all(r.symbol == "bitcoin" for r in records)
    assert all(r.close > 0 for r in records)


@pytest.mark.integration
def test_full_ingestion_agent_live():
    from agents.ingestion.ingestion_agent import IngestionAgent

    agent = IngestionAgent()
    result = agent.execute(
        payload={
            "stock_symbols": ["AAPL"],
            "crypto_symbols": ["bitcoin"],
            "period": "2d",
            "days": 1,
        }
    )
    assert result.succeeded
    total = result.data["summary"]["total_records"]
    print(f"\n  Live records fetched: {total}")
    assert total > 0
