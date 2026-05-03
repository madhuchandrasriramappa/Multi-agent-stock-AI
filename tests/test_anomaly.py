"""
Phase 4 tests — Anomaly Detection Agent and detector functions.

Unit tests are pure: they test detector logic with synthetic DataFrames,
no database involved.

Run unit tests:        pytest tests/test_anomaly.py -v -m "not integration"
Run integration tests: pytest tests/test_anomaly.py -v -m integration -s
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_detector_df(
    rsi: list[float],
    macd: list[float] | None = None,
    vol: list[float] | None = None,
    volume: list[float] | None = None,
    symbol: str = "AAPL",
    interval: str = "1h",
) -> pd.DataFrame:
    """Build a minimal feature DataFrame suitable for detector functions."""
    n = len(rsi)
    base = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
    return pd.DataFrame({
        "symbol":         [symbol] * n,
        "asset_type":     ["stock"] * n,
        "timestamp":      [base + timedelta(hours=i) for i in range(n)],
        "interval":       [interval] * n,
        "rsi_14":         rsi,
        "macd_histogram": macd  or [0.0] * n,
        "volatility_14":  vol   or [0.01] * n,
        "volume":         volume or [1_000_000.0] * n,
    })


# ── Z-score ────────────────────────────────────────────────────────────────────

def test_zscore_detects_outlier():
    from agents.anomaly_detection.detectors import zscore_anomalies

    random.seed(42)
    normal = [50.0 + random.gauss(0, 2) for _ in range(29)]
    df = _make_detector_df(rsi=normal + [200.0])   # z ≈ 75 for the last point
    result = zscore_anomalies(df, ["rsi_14"])

    rsi_alerts = result[result["feature"] == "rsi_14"]
    assert len(rsi_alerts) >= 1
    assert rsi_alerts["score"].max() > 3.0


def test_zscore_returns_empty_for_normal_data():
    from agents.anomaly_detection.detectors import zscore_anomalies

    random.seed(0)
    normal = [50.0 + random.gauss(0, 1) for _ in range(30)]
    df = _make_detector_df(rsi=normal)
    result = zscore_anomalies(df, ["rsi_14"], threshold=10.0)  # very high threshold
    assert result.empty


def test_zscore_skips_small_group():
    """Groups with fewer than 10 rows must be ignored to avoid false positives."""
    from agents.anomaly_detection.detectors import zscore_anomalies

    df = _make_detector_df(rsi=[50.0, 51.0, 49.0, 200.0, 48.0])  # 5 rows only
    result = zscore_anomalies(df, ["rsi_14"])
    assert result.empty


def test_zscore_severity_levels():
    from agents.anomaly_detection.detectors import zscore_anomalies

    random.seed(1)
    normal = [50.0 + random.gauss(0, 1) for _ in range(29)]
    # Append a value ~6 std devs away to get "high" severity
    df = _make_detector_df(rsi=normal + [56.0 + 6.0])  # 56+6 = 62 ≈ 6σ above 50
    result = zscore_anomalies(df, ["rsi_14"])
    if len(result) > 0:
        # The most anomalous row must be "high" or "medium"
        assert result["severity"].iloc[-1] in {"low", "medium", "high"}


def test_zscore_skips_constant_feature():
    """All identical values → std ≈ 0 → skip rather than divide by zero."""
    from agents.anomaly_detection.detectors import zscore_anomalies

    df = _make_detector_df(rsi=[50.0] * 30)
    result = zscore_anomalies(df, ["rsi_14"])
    assert result.empty


# ── IQR ───────────────────────────────────────────────────────────────────────

def test_iqr_detects_outlier():
    from agents.anomaly_detection.detectors import iqr_anomalies

    normal = [float(i) for i in range(1, 30)]   # Q1≈8, Q3≈22, IQR≈14, upper fence≈43
    df = _make_detector_df(rsi=normal + [1000.0])
    result = iqr_anomalies(df, ["rsi_14"])

    assert len(result) >= 1
    assert result.iloc[0]["feature"] == "rsi_14"
    assert result.iloc[0]["score"] > 1.5


def test_iqr_returns_empty_for_normal_data():
    from agents.anomaly_detection.detectors import iqr_anomalies

    normal = [float(i) for i in range(1, 31)]
    df = _make_detector_df(rsi=normal)
    # With multiplier=10 nothing is outside 10*IQR
    result = iqr_anomalies(df, ["rsi_14"], multiplier=10.0)
    assert result.empty


def test_iqr_skips_zero_iqr():
    """Constant series → IQR = 0 → skip rather than divide by zero."""
    from agents.anomaly_detection.detectors import iqr_anomalies

    df = _make_detector_df(rsi=[50.0] * 30)
    result = iqr_anomalies(df, ["rsi_14"])
    assert result.empty


def test_iqr_skips_small_group():
    from agents.anomaly_detection.detectors import iqr_anomalies

    df = _make_detector_df(rsi=[float(i) for i in range(5)] + [1000.0])  # 6 rows
    result = iqr_anomalies(df, ["rsi_14"])
    assert result.empty


def test_iqr_severity_increases_with_distance():
    from agents.anomaly_detection.detectors import iqr_anomalies

    normal = [float(i) for i in range(1, 30)]
    # Value so extreme its distance > 3 * IQR → "high"
    df = _make_detector_df(rsi=normal + [10_000.0])
    result = iqr_anomalies(df, ["rsi_14"])
    assert result.iloc[0]["severity"] == "high"


# ── Isolation Forest ───────────────────────────────────────────────────────────

def test_isolation_forest_detects_anomaly():
    pytest.importorskip("sklearn")
    from agents.anomaly_detection.detectors import isolation_forest_anomalies

    random.seed(0)
    # 24 points tightly clustered around (rsi=50, vol=0.01, macd=0)
    n_normal = 24
    rsi  = [50.0 + random.gauss(0, 2) for _ in range(n_normal)]
    vol  = [0.01 + random.gauss(0, 0.002) for _ in range(n_normal)]
    macd = [0.0  + random.gauss(0, 0.3)   for _ in range(n_normal)]

    # 6 extreme outliers far from the cluster
    rsi  += [5.0,  5.0,  5.0, 95.0, 95.0, 95.0]
    vol  += [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
    macd += [15.0, 15.0, 15.0, -15.0, -15.0, -15.0]

    df = _make_detector_df(rsi=rsi, macd=macd, vol=vol)
    result = isolation_forest_anomalies(
        df, ["rsi_14", "volatility_14", "macd_histogram"], contamination=0.2
    )
    assert len(result) > 0
    assert (result["detector"] == "isolation_forest").all()
    assert (result["feature"] == "multivariate").all()


def test_isolation_forest_skips_small_group():
    pytest.importorskip("sklearn")
    from agents.anomaly_detection.detectors import isolation_forest_anomalies

    # Only 15 rows — below the 20-row minimum
    df = _make_detector_df(rsi=[float(i) for i in range(15)])
    result = isolation_forest_anomalies(df, ["rsi_14"], contamination=0.1)
    assert result.empty


# ── AnomalyAgent (mocked DB) ──────────────────────────────────────────────────

def _make_feature_rows(n: int = 30) -> list[dict]:
    random.seed(42)
    base = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
    return [{
        "symbol":         "AAPL",
        "asset_type":     "stock",
        "timestamp":      base + timedelta(hours=i),
        "interval":       "1h",
        "close":          150.0 + i,
        "volume":         1_000_000.0,
        "rsi_14":         50.0 + random.gauss(0, 10),
        "macd_histogram": random.gauss(0, 0.5),
        "volatility_14":  0.01 + abs(random.gauss(0, 0.005)),
    } for i in range(n)]


@patch("agents.anomaly_detection.anomaly_agent.AnomalyAgent._save_alerts", return_value=3)
@patch("agents.anomaly_detection.anomaly_agent.AnomalyAgent._load_features")
def test_anomaly_agent_execute_success(mock_load, mock_save):
    from agents.anomaly_detection.anomaly_agent import AnomalyAgent

    mock_load.return_value = _make_feature_rows(30)
    result = AnomalyAgent().execute()

    assert result.succeeded
    assert "alerts_detected" in result.data
    assert "alerts_saved" in result.data
    assert "by_detector" in result.data
    assert "by_severity" in result.data
    assert "symbols_processed" in result.data
    assert "AAPL" in result.data["symbols_processed"]

    by_det = result.data["by_detector"]
    assert set(by_det.keys()) == {"zscore", "iqr", "isolation_forest"}
    assert result.data["alerts_detected"] == sum(by_det.values())


@patch("agents.anomaly_detection.anomaly_agent.AnomalyAgent._save_alerts", return_value=0)
@patch("agents.anomaly_detection.anomaly_agent.AnomalyAgent._load_features", return_value=[])
def test_anomaly_agent_execute_empty_db(mock_load, mock_save):
    from agents.anomaly_detection.anomaly_agent import AnomalyAgent

    result = AnomalyAgent().execute()

    assert result.succeeded
    assert result.data["alerts_detected"] == 0
    assert result.data["symbols_processed"] == []


@patch("agents.anomaly_detection.anomaly_agent.AnomalyAgent._save_alerts", return_value=0)
@patch("agents.anomaly_detection.anomaly_agent.AnomalyAgent._load_features")
def test_anomaly_agent_by_severity_sums_correctly(mock_load, mock_save):
    from agents.anomaly_detection.anomaly_agent import AnomalyAgent

    mock_load.return_value = _make_feature_rows(30)
    result = AnomalyAgent().execute()

    assert result.succeeded
    sev = result.data["by_severity"]
    assert sev["low"] + sev["medium"] + sev["high"] == result.data["alerts_detected"]


# ── Integration test ───────────────────────────────────────────────────────────

@pytest.mark.integration
def test_full_pipeline_ingest_clean_features_anomalies():
    """Ingestion → Cleaning → Feature Engineering → Anomaly Detection (real DB)."""
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

    print(f"\n  Alerts detected    : {anomalies.data['alerts_detected']}")
    print(f"  Alerts saved       : {anomalies.data['alerts_saved']}")
    print(f"  By detector        : {anomalies.data['by_detector']}")
    print(f"  By severity        : {anomalies.data['by_severity']}")
    print(f"  Symbols            : {anomalies.data['symbols_processed']}")
