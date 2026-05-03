"""
Anomaly Detection Agent.

Reads feature_set rows (only rows where the key indicators are non-null),
runs three complementary detectors, and writes alerts to anomaly_alerts.

Detectors
---------
Z-score          : univariate; flags values > N std devs from the per-group mean
IQR              : univariate; flags values outside the Tukey fence
Isolation Forest : multivariate; flags points that are easy to isolate in the
                   feature space (low score_samples ↔ anomalous)

Features analysed
-----------------
rsi_14, macd_histogram, volatility_14, volume

Payload keys (all optional)
----------------------------
symbols          : list[str]  — limit to these symbols
since            : str        — ISO-8601 timestamp (load features after this)
limit            : int        — max feature rows to load (default 100_000)
zscore_threshold : float      — |z| cutoff to flag (default 3.0)
iqr_multiplier   : float      — IQR fence multiplier (default 1.5)
contamination    : float      — IF expected outlier fraction (default 0.05)

Output (AgentResult.data)
--------------------------
{
  "alerts_detected"  : int,
  "alerts_saved"     : int,
  "by_detector"      : {"zscore": int, "iqr": int, "isolation_forest": int},
  "by_severity"      : {"low": int, "medium": int, "high": int},
  "symbols_processed": list[str],
}
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import structlog
from sqlalchemy.dialects.postgresql import insert as pg_insert

from agents.anomaly_detection import detectors as det
from agents.base_agent import BaseAgent
from db.connection import get_session
from db.models import AnomalyAlert, FeatureSet

_DEFAULT_LIMIT    = 100_000
_ANOMALY_FEATURES = ["rsi_14", "macd_histogram", "volatility_14", "volume"]


class AnomalyAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "anomaly_agent"

    def run(self, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        p = payload or {}
        symbols          = p.get("symbols")
        since            = p.get("since")
        limit            = int(p.get("limit", _DEFAULT_LIMIT))
        zscore_threshold = float(p.get("zscore_threshold", 3.0))
        iqr_multiplier   = float(p.get("iqr_multiplier", 1.5))
        contamination    = float(p.get("contamination", 0.05))

        rows = self._load_features(symbols=symbols, since=since, limit=limit)
        if not rows:
            self.logger.warning("no_feature_data_found")
            return {
                "alerts_detected":   0,
                "alerts_saved":      0,
                "by_detector":       {"zscore": 0, "iqr": 0, "isolation_forest": 0},
                "by_severity":       {"low": 0, "medium": 0, "high": 0},
                "symbols_processed": [],
            }

        self.logger.info("features_loaded", rows=len(rows))
        df = pd.DataFrame(rows)

        alerts_z   = det.zscore_anomalies(df, _ANOMALY_FEATURES, zscore_threshold)
        alerts_iqr = det.iqr_anomalies(df, _ANOMALY_FEATURES, iqr_multiplier)
        alerts_if  = det.isolation_forest_anomalies(df, _ANOMALY_FEATURES, contamination)

        non_empty = [a for a in [alerts_z, alerts_iqr, alerts_if] if not a.empty]
        all_alerts = pd.concat(non_empty, ignore_index=True) if non_empty else alerts_z

        saved = self._save_alerts(all_alerts)

        def _count(col: str, val: str) -> int:
            return int((all_alerts[col] == val).sum()) if len(all_alerts) else 0

        by_severity = {
            "low":    _count("severity", "low"),
            "medium": _count("severity", "medium"),
            "high":   _count("severity", "high"),
        }

        self.logger.info(
            "anomaly_detection_complete",
            total_alerts=len(all_alerts),
            saved=saved,
            zscore=len(alerts_z),
            iqr=len(alerts_iqr),
            isolation_forest=len(alerts_if),
        )

        return {
            "alerts_detected":   len(all_alerts),
            "alerts_saved":      saved,
            "by_detector": {
                "zscore":           len(alerts_z),
                "iqr":              len(alerts_iqr),
                "isolation_forest": len(alerts_if),
            },
            "by_severity":       by_severity,
            "symbols_processed": sorted(df["symbol"].unique().tolist()),
        }

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _load_features(
        self,
        symbols: list[str] | None = None,
        since: str | None = None,
        limit: int = _DEFAULT_LIMIT,
    ) -> list[dict]:
        with get_session() as session:
            q = session.query(FeatureSet).filter(
                FeatureSet.rsi_14.isnot(None),
                FeatureSet.macd_histogram.isnot(None),
                FeatureSet.volatility_14.isnot(None),
            )
            if symbols:
                q = q.filter(FeatureSet.symbol.in_(symbols))
            if since:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
                q = q.filter(FeatureSet.timestamp >= since_dt)
            q = (q.order_by(FeatureSet.symbol, FeatureSet.interval, FeatureSet.timestamp)
                   .limit(limit))
            rows = q.all()
            return [{
                "symbol":         r.symbol,
                "asset_type":     r.asset_type,
                "timestamp":      r.timestamp,
                "interval":       r.interval,
                "close":          float(r.close),
                "volume":         float(r.volume),
                "rsi_14":         float(r.rsi_14),
                "macd_histogram": float(r.macd_histogram),
                "volatility_14":  float(r.volatility_14),
            } for r in rows]

    def _save_alerts(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        rows = []
        for _, row in df.iterrows():
            rows.append({
                "symbol":        row["symbol"],
                "asset_type":    row["asset_type"],
                "timestamp":     _to_dt(row["timestamp"]),
                "interval":      row["interval"],
                "detector":      row["detector"],
                "feature":       row["feature"],
                "feature_value": _nullable(row.get("feature_value")),
                "score":         float(row["score"]),
                "severity":      row["severity"],
            })

        with get_session() as session:
            stmt = (
                pg_insert(AnomalyAlert)
                .values(rows)
                .on_conflict_do_nothing(
                    index_elements=["symbol", "timestamp", "interval", "detector", "feature"]
                )
            )
            result = session.execute(stmt)
            return result.rowcount if result.rowcount >= 0 else len(rows)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nullable(value) -> float | None:
    if value is None:
        return None
    try:
        return None if math.isnan(float(value)) else round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _to_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if hasattr(value, "to_pydatetime"):
        ts = value.to_pydatetime()
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
