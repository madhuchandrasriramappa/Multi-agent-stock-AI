"""
Phase 3 tests — Feature Engineering Agent and indicator functions.

Unit tests are pure — they test indicator maths with known series,
no database involved.

Run unit tests:        pytest tests/test_features.py -v -m "not integration"
Run integration tests: pytest tests/test_features.py -v -m integration -s
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def _make_ohlcv_df(closes: list[float], base_ts: datetime | None = None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for testing VWAP and the full agent."""
    if base_ts is None:
        base_ts = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
    n = len(closes)
    timestamps = [base_ts + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame({
        "symbol":     ["AAPL"] * n,
        "asset_type": ["stock"] * n,
        "timestamp":  timestamps,
        "interval":   ["1h"] * n,
        "open":       [c - 0.5 for c in closes],
        "high":       [c + 1.0 for c in closes],
        "low":        [c - 1.0 for c in closes],
        "close":      closes,
        "volume":     [1_000_000.0] * n,
    })


# ── SMA ────────────────────────────────────────────────────────────────────────

def test_sma_basic():
    from agents.feature_engineering.indicators import sma
    s = _make_series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = sma(s, window=3)
    # First 2 rows are NaN (insufficient window)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    # Third row: avg(1,2,3) = 2
    assert result.iloc[2] == pytest.approx(2.0)
    # Last row: avg(8,9,10) = 9
    assert result.iloc[-1] == pytest.approx(9.0)


def test_sma_window_larger_than_series():
    from agents.feature_engineering.indicators import sma
    result = sma(_make_series([1, 2, 3]), window=10)
    assert result.isna().all()


def test_sma_constant_series():
    from agents.feature_engineering.indicators import sma
    result = sma(_make_series([5.0] * 30), window=20)
    assert (result.dropna().sub(5.0).abs() < 1e-6).all()


# ── EMA ────────────────────────────────────────────────────────────────────────

def test_ema_no_nans():
    """EMA seeds from first value so it should never produce NaN."""
    from agents.feature_engineering.indicators import ema
    result = ema(_make_series([10, 11, 12, 13, 14]), span=3)
    assert result.notna().all()


def test_ema_reacts_faster_than_sma():
    """EMA should be closer to recent prices than SMA after an upward jump."""
    from agents.feature_engineering.indicators import ema, sma
    # Flat at 10 for 20 periods, then jump to 50
    values = [10.0] * 20 + [50.0] * 5
    s = _make_series(values)
    ema_val = ema(s, 12).iloc[-1]
    sma_val = sma(s, 12).iloc[-1]
    assert ema_val > sma_val   # EMA adapts faster to the jump


# ── RSI ────────────────────────────────────────────────────────────────────────

def test_rsi_strictly_increasing_is_100():
    from agents.feature_engineering.indicators import rsi
    s = _make_series(list(range(1, 50)))   # always going up
    result = rsi(s, period=14)
    non_null = result.dropna()
    assert (non_null == pytest.approx(100.0)).all()


def test_rsi_strictly_decreasing_is_0():
    from agents.feature_engineering.indicators import rsi
    s = _make_series(list(range(50, 1, -1)))   # always going down
    result = rsi(s, period=14)
    assert (result.dropna().abs() < 1e-6).all()


def test_rsi_in_valid_range():
    from agents.feature_engineering.indicators import rsi
    import random
    random.seed(42)
    values = [100 + random.gauss(0, 5) for _ in range(100)]
    result = rsi(_make_series(values), period=14)
    non_null = result.dropna()
    assert (non_null >= 0).all() and (non_null <= 100).all()


def test_rsi_first_n_rows_are_nan():
    from agents.feature_engineering.indicators import rsi
    result = rsi(_make_series(list(range(1, 50))), period=14)
    # First 14 rows (diffs) + 1 (diff itself creates NaN at 0) need to be NaN
    assert result.iloc[:14].isna().all()


# ── Bollinger Bands ────────────────────────────────────────────────────────────

def test_bollinger_bands_structure():
    from agents.feature_engineering.indicators import bollinger_bands
    s = _make_series([float(i) for i in range(1, 51)])
    upper, middle, lower = bollinger_bands(s, window=20)
    non_null_idx = middle.dropna().index
    # upper > middle > lower always
    assert (upper.loc[non_null_idx] >= middle.loc[non_null_idx]).all()
    assert (middle.loc[non_null_idx] >= lower.loc[non_null_idx]).all()


def test_bollinger_bands_middle_equals_sma():
    from agents.feature_engineering.indicators import bollinger_bands, sma
    s = _make_series([float(i) for i in range(1, 51)])
    _, middle, _ = bollinger_bands(s, window=20)
    expected = sma(s, 20)
    pd.testing.assert_series_equal(middle, expected, check_names=False)


def test_bollinger_bands_constant_series_zero_width():
    from agents.feature_engineering.indicators import bollinger_bands
    s = _make_series([10.0] * 30)
    upper, middle, lower = bollinger_bands(s, window=20)
    idx = middle.dropna().index
    assert (upper.loc[idx].sub(10.0).abs() < 1e-6).all()
    assert (lower.loc[idx].sub(10.0).abs() < 1e-6).all()


# ── MACD ───────────────────────────────────────────────────────────────────────

def test_macd_line_equals_ema_diff():
    from agents.feature_engineering.indicators import ema, macd
    s = _make_series([float(i) + 0.1 * (i % 3) for i in range(1, 60)])
    macd_line, _, _ = macd(s, fast=12, slow=26)
    expected = ema(s, 12) - ema(s, 26)
    pd.testing.assert_series_equal(macd_line, expected, check_names=False)


def test_macd_histogram_equals_line_minus_signal():
    from agents.feature_engineering.indicators import macd
    s = _make_series([float(i) for i in range(1, 60)])
    line, signal, hist = macd(s)
    expected_hist = line - signal
    pd.testing.assert_series_equal(hist, expected_hist, check_names=False)


# ── Volatility ─────────────────────────────────────────────────────────────────

def test_volatility_constant_series_is_zero():
    from agents.feature_engineering.indicators import rolling_volatility
    s = _make_series([100.0] * 30)
    result = rolling_volatility(s, window=14)
    assert (result.dropna().abs() < 1e-10).all()


def test_volatility_non_negative():
    from agents.feature_engineering.indicators import rolling_volatility
    import random
    random.seed(0)
    s = _make_series([100 + random.gauss(0, 2) for _ in range(50)])
    result = rolling_volatility(s, window=14)
    assert (result.dropna() >= 0).all()


# ── VWAP ───────────────────────────────────────────────────────────────────────

def test_vwap_equals_typical_price_with_equal_volume():
    """When all candles have equal volume, VWAP = mean of typical prices."""
    from agents.feature_engineering.indicators import vwap_daily
    df = _make_ohlcv_df([100.0, 102.0, 104.0, 106.0])
    result = vwap_daily(df)
    # All on the same day with equal volume, so VWAP is cumulative avg of typical prices
    assert result.notna().all()
    # First candle: VWAP = typical_price = (101+99+100)/3 = 100
    expected_tp_0 = (df.iloc[0]["high"] + df.iloc[0]["low"] + df.iloc[0]["close"]) / 3
    assert result.iloc[0] == pytest.approx(expected_tp_0)


def test_vwap_resets_on_new_day():
    from agents.feature_engineering.indicators import vwap_daily
    # 2 candles on day 1, 2 candles on day 2
    base = datetime(2024, 1, 15, 22, 0, 0, tzinfo=timezone.utc)
    df = _make_ohlcv_df([100.0, 101.0, 200.0, 201.0], base_ts=base)
    result = vwap_daily(df)
    # Day 2 VWAP should be near 200, not influenced by day 1's prices
    assert result.iloc[2] == pytest.approx(
        (df.iloc[2]["high"] + df.iloc[2]["low"] + df.iloc[2]["close"]) / 3,
        rel=1e-3
    )


# ── Feature Agent (_compute_features) ─────────────────────────────────────────

def test_feature_agent_compute_features_columns():
    from agents.feature_engineering.feature_agent import FeatureAgent
    agent = FeatureAgent()
    df = _make_ohlcv_df([float(100 + i) for i in range(60)])
    result = agent._compute_features(df)

    expected_cols = {
        "symbol", "asset_type", "timestamp", "interval", "close", "volume",
        "sma_20", "sma_50", "ema_12", "ema_26",
        "rsi_14", "macd_line", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower",
        "volatility_14", "vwap",
    }
    assert expected_cols.issubset(set(result.columns))


def test_feature_agent_compute_features_row_count_preserved():
    from agents.feature_engineering.feature_agent import FeatureAgent
    agent = FeatureAgent()
    df = _make_ohlcv_df([float(100 + i) for i in range(30)])
    result = agent._compute_features(df)
    assert len(result) == 30


def test_feature_agent_early_rows_have_nulls():
    """With only 30 rows, SMA-50 must be all null."""
    from agents.feature_engineering.feature_agent import FeatureAgent
    agent = FeatureAgent()
    df = _make_ohlcv_df([float(100 + i) for i in range(30)])
    result = agent._compute_features(df)
    assert result["sma_50"].isna().all()


def test_feature_agent_sufficient_rows_fills_sma20():
    """With 25+ rows, at least the last 6 rows of SMA-20 should be non-null."""
    from agents.feature_engineering.feature_agent import FeatureAgent
    agent = FeatureAgent()
    df = _make_ohlcv_df([float(100 + i) for i in range(25)])
    result = agent._compute_features(df)
    assert result["sma_20"].notna().sum() >= 6


# ── Full agent execute() (mocked DB) ──────────────────────────────────────────

def _make_clean_row(i: int) -> dict:
    now = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
    return {
        "symbol":     "AAPL",
        "asset_type": "stock",
        "timestamp":  now + timedelta(hours=i),
        "interval":   "1h",
        "open":       100.0 + i - 0.5,
        "high":       100.0 + i + 1.0,
        "low":        100.0 + i - 1.0,
        "close":      float(100 + i),
        "volume":     1_000_000.0,
    }


@patch("agents.feature_engineering.feature_agent.FeatureAgent._save_features", return_value=30)
@patch("agents.feature_engineering.feature_agent.FeatureAgent._load_clean_data")
def test_feature_agent_execute_success(mock_load, mock_save):
    from agents.feature_engineering.feature_agent import FeatureAgent

    mock_load.return_value = [_make_clean_row(i) for i in range(30)]

    result = FeatureAgent().execute()

    assert result.succeeded
    assert result.data["features_computed"] == 30
    assert result.data["features_saved"] == 30
    assert "AAPL" in result.data["symbols_processed"]


@patch("agents.feature_engineering.feature_agent.FeatureAgent._save_features", return_value=0)
@patch("agents.feature_engineering.feature_agent.FeatureAgent._load_clean_data", return_value=[])
def test_feature_agent_execute_empty_db(mock_load, mock_save):
    from agents.feature_engineering.feature_agent import FeatureAgent

    result = FeatureAgent().execute()

    assert result.succeeded
    assert result.data["features_computed"] == 0


# ── Integration test ───────────────────────────────────────────────────────────

@pytest.mark.integration
def test_full_pipeline_ingest_clean_features():
    """Ingestion → Cleaning → Feature Engineering against real local PostgreSQL."""
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

    print(f"\n  Features computed  : {features.data['features_computed']}")
    print(f"  Features saved     : {features.data['features_saved']}")
    print(f"  Rows with full data: {features.data['rows_with_full_data']}")
    print(f"  Rows with nulls    : {features.data['rows_with_nulls']}")
    print(f"  Symbols            : {features.data['symbols_processed']}")

    assert features.data["features_computed"] > 0
