"""
Feature Engineering Agent.

Reads cleaned OHLCV data from clean_market_data, computes a suite of
technical indicators for every (symbol, interval) group, and writes
the results to the feature_set table.

Indicators computed
-------------------
SMA 20, SMA 50           — trend direction
EMA 12, EMA 26           — faster-reacting trend
RSI 14                   — overbought / oversold (0-100)
MACD line, signal, hist  — momentum crossover signals
Bollinger upper/mid/low  — volatility envelope
Volatility 14            — rolling std of returns
VWAP                     — volume-weighted fair value (resets daily)

NaN policy
----------
Indicators that require more history than exists (e.g. SMA-50 on 30 rows)
produce NaN for early rows.  These are stored as NULL in the DB.
Phase 4 (Anomaly Detection) filters them out before analysis.

Payload keys (all optional)
----------------------------
symbols  : list[str]  — limit to these symbols (default: all in clean table)
since    : str        — ISO-8601 timestamp, load data after this point
limit    : int        — max clean rows to load (default 100_000)

Output (AgentResult.data)
--------------------------
{
  "features_computed"  : int,   total rows in the result DataFrame
  "features_saved"     : int,   rows written to DB (skips existing)
  "symbols_processed"  : list[str],
  "rows_with_full_data": int,   rows where every indicator is non-null
  "rows_with_nulls"    : int,   rows with at least one null (insufficient history)
}
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import structlog
from sqlalchemy.dialects.postgresql import insert as pg_insert

from agents.base_agent import BaseAgent
from agents.feature_engineering import indicators as ind
from db.connection import get_session
from db.models import CleanMarketData, FeatureSet

_DEFAULT_LIMIT = 100_000


class FeatureAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "feature_agent"

    def run(self, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        p = payload or {}
        symbols: list[str] | None = p.get("symbols")
        since:   str | None       = p.get("since")
        limit:   int              = int(p.get("limit", _DEFAULT_LIMIT))

        # ── Load clean data ────────────────────────────────────────────────────
        rows = self._load_clean_data(symbols=symbols, since=since, limit=limit)
        if not rows:
            self.logger.warning("no_clean_data_found")
            return {
                "features_computed":   0,
                "features_saved":      0,
                "symbols_processed":   [],
                "rows_with_full_data": 0,
                "rows_with_nulls":     0,
            }

        self.logger.info("clean_data_loaded", rows=len(rows))

        # ── Compute features per (symbol, interval) group ─────────────────────
        df_raw = pd.DataFrame(rows)

        feature_frames: list[pd.DataFrame] = []
        for (symbol, interval), group in df_raw.groupby(["symbol", "interval"]):
            group = group.sort_values("timestamp").reset_index(drop=True)
            self.logger.info("computing_features", symbol=symbol, interval=interval, rows=len(group))
            feature_frames.append(self._compute_features(group))

        all_features = pd.concat(feature_frames, ignore_index=True)

        # ── Save ───────────────────────────────────────────────────────────────
        saved = self._save_features(all_features)

        indicator_cols = [
            "sma_20", "sma_50", "ema_12", "ema_26",
            "rsi_14", "macd_line", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower",
            "volatility_14", "vwap",
        ]
        rows_full = int(all_features[indicator_cols].notna().all(axis=1).sum())
        rows_null = len(all_features) - rows_full

        self.logger.info(
            "features_complete",
            total=len(all_features),
            saved=saved,
            full_rows=rows_full,
            null_rows=rows_null,
        )

        return {
            "features_computed":   len(all_features),
            "features_saved":      saved,
            "symbols_processed":   sorted(all_features["symbol"].unique().tolist()),
            "rows_with_full_data": rows_full,
            "rows_with_nulls":     rows_null,
        }

    # ── Core computation ───────────────────────────────────────────────────────

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators for a single (symbol, interval) DataFrame.
        df must be sorted by timestamp ascending with a clean integer index.
        """
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        result = df[["symbol", "asset_type", "timestamp", "interval", "close", "volume"]].copy()

        # Moving averages
        result["sma_20"] = ind.sma(close, 20)
        result["sma_50"] = ind.sma(close, 50)
        result["ema_12"] = ind.ema(close, 12)
        result["ema_26"] = ind.ema(close, 26)

        # RSI
        result["rsi_14"] = ind.rsi(close, 14)

        # MACD
        result["macd_line"], result["macd_signal"], result["macd_histogram"] = ind.macd(close)

        # Bollinger Bands
        result["bb_upper"], result["bb_middle"], result["bb_lower"] = ind.bollinger_bands(close)

        # Volatility
        result["volatility_14"] = ind.rolling_volatility(close, 14)

        # VWAP (needs high/low/close/volume/timestamp — pass the full df)
        result["vwap"] = ind.vwap_daily(df)

        return result

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _load_clean_data(
        self,
        symbols: list[str] | None = None,
        since: str | None = None,
        limit: int = _DEFAULT_LIMIT,
    ) -> list[dict]:
        with get_session() as session:
            q = session.query(CleanMarketData)
            if symbols:
                q = q.filter(CleanMarketData.symbol.in_(symbols))
            if since:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
                q = q.filter(CleanMarketData.timestamp >= since_dt)
            q = (q.order_by(CleanMarketData.symbol,
                            CleanMarketData.interval,
                            CleanMarketData.timestamp)
                   .limit(limit))
            rows = q.all()
            return [{
                "symbol":     r.symbol,
                "asset_type": r.asset_type,
                "timestamp":  r.timestamp,
                "interval":   r.interval,
                "open":       float(r.open),
                "high":       float(r.high),
                "low":        float(r.low),
                "close":      float(r.close),
                "volume":     float(r.volume),
            } for r in rows]

    def _save_features(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        rows = []
        for _, row in df.iterrows():
            rows.append({
                "symbol":          row["symbol"],
                "asset_type":      row["asset_type"],
                "timestamp":       _to_dt(row["timestamp"]),
                "interval":        row["interval"],
                "close":           float(row["close"]),
                "volume":          float(row["volume"]),
                "sma_20":          _nullable(row.get("sma_20")),
                "sma_50":          _nullable(row.get("sma_50")),
                "ema_12":          _nullable(row.get("ema_12")),
                "ema_26":          _nullable(row.get("ema_26")),
                "rsi_14":          _nullable(row.get("rsi_14")),
                "macd_line":       _nullable(row.get("macd_line")),
                "macd_signal":     _nullable(row.get("macd_signal")),
                "macd_histogram":  _nullable(row.get("macd_histogram")),
                "bb_upper":        _nullable(row.get("bb_upper")),
                "bb_middle":       _nullable(row.get("bb_middle")),
                "bb_lower":        _nullable(row.get("bb_lower")),
                "volatility_14":   _nullable(row.get("volatility_14")),
                "vwap":            _nullable(row.get("vwap")),
            })

        with get_session() as session:
            stmt = (
                pg_insert(FeatureSet)
                .values(rows)
                .on_conflict_do_nothing(
                    index_elements=["symbol", "timestamp", "interval"]
                )
            )
            result = session.execute(stmt)
            return result.rowcount if result.rowcount >= 0 else len(rows)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nullable(value) -> float | None:
    """Convert NaN / None to Python None so SQLAlchemy stores NULL."""
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
