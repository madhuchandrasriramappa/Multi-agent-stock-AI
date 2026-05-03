"""
Phase 2 tests — Data Cleaning Agent.

All unit tests are pure — they test the _clean() method directly
with DataFrames, no database involved.

Run:  pytest tests/test_cleaning.py -v
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_record(
    symbol="AAPL",
    asset_type="stock",
    ts_offset_hours=0,
    open=182.0,
    high=185.0,
    low=181.0,
    close=184.0,
    volume=5_000_000.0,
    source="yahoo_finance",
    interval="1h",
    **kwargs,
) -> dict:
    now = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
    ts = now + timedelta(hours=ts_offset_hours)
    base = dict(
        symbol=symbol, asset_type=asset_type, timestamp=ts.isoformat(),
        open=open, high=high, low=low, close=close,
        volume=volume, source=source, fetched_at=ts.isoformat(), interval=interval,
    )
    base.update(kwargs)
    return base


def _make_df(*records) -> pd.DataFrame:
    return pd.DataFrame(list(records))


def _get_agent():
    from agents.cleaning.cleaning_agent import CleaningAgent
    return CleaningAgent()


# ── Deduplication ──────────────────────────────────────────────────────────────

def test_clean_removes_exact_duplicates():
    agent = _get_agent()
    record = _make_record()
    df = _make_df(record, record, record)           # 3 identical rows

    df_clean, stats = agent._clean(df)

    assert len(df_clean) == 1
    assert stats["drop_reasons"].get("duplicates") == 2


def test_clean_keeps_same_symbol_different_timestamps():
    agent = _get_agent()
    df = _make_df(
        _make_record(ts_offset_hours=0),
        _make_record(ts_offset_hours=1),
        _make_record(ts_offset_hours=2),
    )
    df_clean, stats = agent._clean(df)

    assert len(df_clean) == 3
    assert "duplicates" not in stats["drop_reasons"]


# ── NaN price handling ─────────────────────────────────────────────────────────

def test_clean_drops_nan_close():
    agent = _get_agent()
    bad = _make_record(close=float("nan"))
    good = _make_record(ts_offset_hours=1)
    df_clean, stats = agent._clean(_make_df(bad, good))

    assert len(df_clean) == 1
    assert stats["drop_reasons"].get("nan_prices") == 1


def test_clean_drops_nan_open():
    agent = _get_agent()
    bad = _make_record(open=float("nan"))
    df_clean, stats = agent._clean(_make_df(bad))

    assert len(df_clean) == 0
    assert stats["drop_reasons"].get("nan_prices") == 1


def test_clean_fills_nan_volume_with_zero():
    agent = _get_agent()
    record = _make_record(volume=float("nan"))
    df_clean, _ = agent._clean(_make_df(record))

    assert len(df_clean) == 1
    assert df_clean.iloc[0]["volume"] == 0.0


# ── Non-positive prices ────────────────────────────────────────────────────────

def test_clean_drops_zero_close():
    agent = _get_agent()
    df_clean, stats = agent._clean(_make_df(_make_record(close=0.0)))

    assert len(df_clean) == 0
    assert stats["drop_reasons"].get("non_positive_prices") == 1


def test_clean_drops_negative_price():
    agent = _get_agent()
    df_clean, stats = agent._clean(_make_df(_make_record(open=-1.0)))

    assert len(df_clean) == 0
    assert "non_positive_prices" in stats["drop_reasons"]


# ── OHLC logic validation ──────────────────────────────────────────────────────

def test_clean_drops_high_less_than_low():
    agent = _get_agent()
    bad = _make_record(high=180.0, low=185.0)   # high < low → invalid
    df_clean, stats = agent._clean(_make_df(bad))

    assert len(df_clean) == 0
    assert stats["drop_reasons"].get("invalid_ohlc") == 1


def test_clean_drops_high_less_than_close():
    agent = _get_agent()
    bad = _make_record(high=183.0, close=184.0)  # close > high → impossible
    df_clean, stats = agent._clean(_make_df(bad))

    assert len(df_clean) == 0
    assert "invalid_ohlc" in stats["drop_reasons"]


def test_clean_keeps_valid_ohlc():
    agent = _get_agent()
    good = _make_record(open=182.0, high=186.0, low=181.0, close=184.0)
    df_clean, stats = agent._clean(_make_df(good))

    assert len(df_clean) == 1
    assert "invalid_ohlc" not in stats["drop_reasons"]


# ── Future timestamp ───────────────────────────────────────────────────────────

def test_clean_drops_future_timestamp():
    agent = _get_agent()
    from datetime import timedelta
    future_ts = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
    bad = _make_record()
    bad["timestamp"] = future_ts

    df_clean, stats = agent._clean(_make_df(bad))

    assert len(df_clean) == 0
    assert stats["drop_reasons"].get("future_timestamps") == 1


# ── Outlier flagging ───────────────────────────────────────────────────────────

def test_clean_flags_large_price_move_as_outlier():
    agent = _get_agent()
    # Row 0: close=100, Row 1: close=130 → 30% jump → outlier
    # All OHLC values must be internally consistent
    records = [
        _make_record(ts_offset_hours=0, open=99.0,  high=101.0, low=98.0,  close=100.0),
        _make_record(ts_offset_hours=1, open=128.0, high=131.0, low=127.0, close=130.0),
    ]
    df_clean, stats = agent._clean(_make_df(*records))

    assert len(df_clean) == 2                     # row is KEPT, not dropped
    assert stats["outliers_flagged"] == 1
    outlier_row = df_clean[df_clean["is_outlier"] == True]
    assert len(outlier_row) == 1
    assert outlier_row.iloc[0]["close"] == 130.0
    assert "cleaning_notes" in outlier_row.columns


def test_clean_does_not_flag_normal_move():
    agent = _get_agent()
    records = [
        _make_record(ts_offset_hours=0, close=100.0, high=101.0),
        _make_record(ts_offset_hours=1, close=101.5, high=102.0, low=100.0),  # 1.5%
    ]
    df_clean, stats = agent._clean(_make_df(*records))

    assert stats["outliers_flagged"] == 0
    assert df_clean["is_outlier"].sum() == 0


# ── Stats structure ────────────────────────────────────────────────────────────

def test_clean_returns_correct_stats_structure():
    agent = _get_agent()
    df_clean, stats = agent._clean(_make_df(_make_record()))

    assert "dropped_total" in stats
    assert "drop_reasons" in stats
    assert "outliers_flagged" in stats
    assert isinstance(stats["drop_reasons"], dict)


def test_clean_empty_dataframe():
    agent = _get_agent()
    df_clean, stats = agent._clean(pd.DataFrame())

    assert df_clean.empty
    assert stats["dropped_total"] == 0


# ── Full agent (mocked DB) ─────────────────────────────────────────────────────

@patch("agents.cleaning.cleaning_agent.CleaningAgent._save_raw", return_value=5)
@patch("agents.cleaning.cleaning_agent.CleaningAgent._save_clean", return_value=4)
def test_cleaning_agent_execute_returns_agent_result(mock_save_clean, mock_save_raw):
    from agents.cleaning.cleaning_agent import CleaningAgent

    records = [_make_record(ts_offset_hours=i) for i in range(5)]
    agent = CleaningAgent()
    result = agent.execute(payload={"records": records})

    assert result.succeeded
    assert result.agent == "cleaning_agent"
    assert result.data["raw_records_saved"] == 5
    assert result.data["clean_records_saved"] == 4
    assert isinstance(result.data["symbols_processed"], list)


@patch("agents.cleaning.cleaning_agent.CleaningAgent._save_raw", return_value=0)
@patch("agents.cleaning.cleaning_agent.CleaningAgent._save_clean", return_value=0)
def test_cleaning_agent_handles_empty_payload(mock_save_clean, mock_save_raw):
    from agents.cleaning.cleaning_agent import CleaningAgent

    result = CleaningAgent().execute(payload={"records": []})

    assert result.succeeded
    assert result.data["clean_records_saved"] == 0


# ── Integration test (real DB) ─────────────────────────────────────────────────

@pytest.mark.integration
def test_full_pipeline_ingest_then_clean():
    """Runs ingestion → cleaning against real local PostgreSQL."""
    from agents.cleaning.cleaning_agent import CleaningAgent
    from agents.ingestion.ingestion_agent import IngestionAgent
    from db.migrate import run_migrations

    run_migrations()

    ingest_result = IngestionAgent().execute(payload={
        "stock_symbols": ["AAPL"],
        "crypto_symbols": ["bitcoin"],
        "period": "2d",
        "days": 1,
    })
    assert ingest_result.succeeded

    all_records = (
        ingest_result.data["stocks"] + ingest_result.data["crypto"]
    )
    clean_result = CleaningAgent().execute(payload={"records": all_records})

    assert clean_result.succeeded
    assert clean_result.data["clean_records_saved"] > 0
    print(f"\n  Records cleaned and saved: {clean_result.data['clean_records_saved']}")
    print(f"  Dropped: {clean_result.data['dropped_total']}")
    print(f"  Outliers flagged: {clean_result.data['outliers_flagged']}")
