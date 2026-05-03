"""
Data Cleaning Agent.

Responsibilities
----------------
1. Persist raw records from the Ingestion Agent into raw_market_data
2. Apply cleaning rules to produce a validated dataset
3. Write cleaned records into clean_market_data

Cleaning rules applied (in order)
----------------------------------
1. Deduplication         — same (symbol, timestamp, source, interval) → keep first
2. NaN prices            — any of open/high/low/close is NaN → drop row
3. Volume NaN            — fill with 0 (volume missing is common in crypto off-hours)
4. Non-positive prices   — open/high/low/close <= 0 → drop (data error)
5. OHLC logic            — high < low or high < open/close etc → drop (corrupt candle)
6. Future timestamps     — timestamp > now(UTC) → drop
7. Outlier flagging      — abs(pct_change in close) > 20% per candle → is_outlier=True
                           Row is kept; downstream agents decide how to handle it.

Input (payload keys — all optional)
------------------------------------
records        : list[dict]   Raw MarketRecord dicts from the Ingestion Agent.
                               If omitted, the agent reads directly from raw_market_data.
symbols        : list[str]    Filter which symbols to clean (only used in DB-read mode).
limit          : int          Max raw rows to load from DB per run (default 10_000).

Output (AgentResult.data)
--------------------------
{
  "raw_records_saved"   : int,
  "clean_records_saved" : int,
  "dropped_total"       : int,
  "drop_reasons"        : dict,   # {"duplicates": 3, "nan_prices": 1, ...}
  "outliers_flagged"    : int,
  "symbols_processed"   : list[str],
}
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import structlog
from sqlalchemy.dialects.postgresql import insert as pg_insert

from agents.base_agent import BaseAgent
from db.connection import get_session
from db.models import CleanMarketData, RawMarketData

logger = structlog.get_logger(agent="cleaning_agent")

_OUTLIER_THRESHOLD = 0.20   # 20% single-candle price move triggers outlier flag
_DEFAULT_DB_LIMIT  = 10_000


class CleaningAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "cleaning_agent"

    def run(self, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        p = payload or {}
        records_in_payload = "records" in p
        raw_records: list[dict] = p.get("records", [])
        symbols_filter: list[str] | None = p.get("symbols")
        limit: int = int(p.get("limit", _DEFAULT_DB_LIMIT))

        # ── Step 1: source raw records ─────────────────────────────────────────
        raw_saved = 0
        if records_in_payload:
            # Caller passed records directly (pipeline mode)
            if raw_records:
                raw_saved = self._save_raw(raw_records)
                self.logger.info("raw_records_saved", count=raw_saved)
        else:
            # Standalone mode: load uncleaned records from DB
            raw_records = self._load_raw_from_db(symbols=symbols_filter, limit=limit)
            self.logger.info("raw_records_loaded_from_db", count=len(raw_records))

        if not raw_records:
            self.logger.warning("no_records_to_clean")
            return {
                "raw_records_saved": 0,
                "clean_records_saved": 0,
                "dropped_total": 0,
                "drop_reasons": {},
                "outliers_flagged": 0,
                "symbols_processed": [],
            }

        # ── Step 2: clean ──────────────────────────────────────────────────────
        df_clean, stats = self._clean(pd.DataFrame(raw_records))

        # ── Step 3: persist clean records ─────────────────────────────────────
        clean_saved = self._save_clean(df_clean)
        self.logger.info("clean_records_saved", count=clean_saved)

        symbols_processed = sorted(df_clean["symbol"].unique().tolist()) if not df_clean.empty else []

        return {
            "raw_records_saved":   raw_saved,
            "clean_records_saved": clean_saved,
            "dropped_total":       stats["dropped_total"],
            "drop_reasons":        stats["drop_reasons"],
            "outliers_flagged":    stats["outliers_flagged"],
            "symbols_processed":   symbols_processed,
        }

    # ── Cleaning logic ─────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Apply all cleaning rules to a DataFrame of raw records.
        Returns (cleaned_df, stats_dict).
        Pure function — no DB access.
        """
        if df.empty:
            return df, {"dropped_total": 0, "drop_reasons": {}, "outliers_flagged": 0}

        initial = len(df)
        drop_reasons: dict[str, int] = {}

        # Normalise timestamp to UTC datetime for consistent comparison
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        # 1. Deduplication ─────────────────────────────────────────────────────
        before = len(df)
        df = df.drop_duplicates(subset=["symbol", "timestamp", "source", "interval"], keep="first")
        _record_drop(drop_reasons, "duplicates", before - len(df))

        # 2. NaN prices ────────────────────────────────────────────────────────
        before = len(df)
        df = df.dropna(subset=["open", "high", "low", "close"])
        _record_drop(drop_reasons, "nan_prices", before - len(df))

        # 3. Volume NaN → 0 (not a drop, just a fill) ─────────────────────────
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

        # 4. Non-positive prices ───────────────────────────────────────────────
        before = len(df)
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        valid_prices = (df[price_cols] > 0).all(axis=1)
        df = df[valid_prices]
        _record_drop(drop_reasons, "non_positive_prices", before - len(df))

        # 5. OHLC logic ────────────────────────────────────────────────────────
        before = len(df)
        valid_ohlc = (
            (df["high"] >= df["low"])   &
            (df["high"] >= df["open"])  &
            (df["high"] >= df["close"]) &
            (df["low"]  <= df["open"])  &
            (df["low"]  <= df["close"])
        )
        df = df[valid_ohlc]
        _record_drop(drop_reasons, "invalid_ohlc", before - len(df))

        # 6. Future timestamps ─────────────────────────────────────────────────
        before = len(df)
        now_utc = pd.Timestamp.now(tz="UTC")
        df = df[df["timestamp"] <= now_utc]
        _record_drop(drop_reasons, "future_timestamps", before - len(df))

        # 7. Outlier flagging (soft — row kept, is_outlier=True) ───────────────
        df["is_outlier"]     = False
        df["cleaning_notes"] = None

        for (symbol, interval), grp in df.groupby(["symbol", "interval"]):
            pct_chg = grp["close"].pct_change().abs()
            outlier_idx = pct_chg[pct_chg > _OUTLIER_THRESHOLD].index
            if len(outlier_idx):
                df.loc[outlier_idx, "is_outlier"]     = True
                df.loc[outlier_idx, "cleaning_notes"] = (
                    f"single_candle_close_change_exceeds_{int(_OUTLIER_THRESHOLD*100)}pct"
                )

        outliers_flagged = int(df["is_outlier"].sum())
        dropped_total    = initial - len(df)

        self.logger.info(
            "cleaning_complete",
            initial=initial,
            remaining=len(df),
            dropped=dropped_total,
            outliers_flagged=outliers_flagged,
            drop_reasons=drop_reasons,
        )

        return df, {
            "dropped_total":    dropped_total,
            "drop_reasons":     drop_reasons,
            "outliers_flagged": outliers_flagged,
        }

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _save_raw(self, records: list[dict]) -> int:
        """
        Upsert raw records into raw_market_data.
        Rows that already exist (same unique key) are silently skipped.
        Returns number of rows actually inserted.
        """
        if not records:
            return 0

        rows = [_to_raw_row(r) for r in records]

        with get_session() as session:
            stmt = (
                pg_insert(RawMarketData)
                .values(rows)
                .on_conflict_do_nothing(
                    index_elements=["symbol", "timestamp", "source", "interval"]
                )
            )
            result = session.execute(stmt)
            return result.rowcount if result.rowcount >= 0 else len(rows)

    def _load_raw_from_db(
        self,
        symbols: list[str] | None = None,
        limit: int = _DEFAULT_DB_LIMIT,
    ) -> list[dict]:
        """Load raw records from the DB for standalone cleaning runs."""
        with get_session() as session:
            q = session.query(RawMarketData)
            if symbols:
                q = q.filter(RawMarketData.symbol.in_(symbols))
            q = q.order_by(RawMarketData.symbol, RawMarketData.timestamp).limit(limit)
            rows = q.all()

        return [
            {
                "symbol":     r.symbol,
                "asset_type": r.asset_type,
                "timestamp":  r.timestamp,
                "open":       float(r.open),
                "high":       float(r.high),
                "low":        float(r.low),
                "close":      float(r.close),
                "volume":     float(r.volume),
                "source":     r.source,
                "fetched_at": r.fetched_at,
                "interval":   r.interval,
            }
            for r in rows
        ]

    def _save_clean(self, df: pd.DataFrame) -> int:
        """
        Upsert cleaned records into clean_market_data.
        Returns number of rows actually inserted.
        """
        if df.empty:
            return 0

        rows = [_to_clean_row(row) for _, row in df.iterrows()]

        with get_session() as session:
            stmt = (
                pg_insert(CleanMarketData)
                .values(rows)
                .on_conflict_do_nothing(
                    index_elements=["symbol", "timestamp", "source", "interval"]
                )
            )
            result = session.execute(stmt)
            return result.rowcount if result.rowcount >= 0 else len(rows)


# ── Module-level helpers ───────────────────────────────────────────────────────

def _record_drop(reasons: dict[str, int], key: str, count: int) -> None:
    if count > 0:
        reasons[key] = count


def _to_raw_row(r: dict) -> dict:
    """Convert a MarketRecord dict to a RawMarketData column dict."""
    return {
        "symbol":     r["symbol"],
        "asset_type": r["asset_type"],
        "timestamp":  _parse_ts(r["timestamp"]),
        "open":       float(r["open"]),
        "high":       float(r["high"]),
        "low":        float(r["low"]),
        "close":      float(r["close"]),
        "volume":     float(r.get("volume") or 0),
        "source":     r["source"],
        "fetched_at": _parse_ts(r["fetched_at"]),
        "interval":   r["interval"],
    }


def _to_clean_row(row: "pd.Series") -> dict:
    """Convert a cleaned DataFrame row to a CleanMarketData column dict."""
    return {
        "symbol":         row["symbol"],
        "asset_type":     row["asset_type"],
        "timestamp":      row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
        "open":           float(row["open"]),
        "high":           float(row["high"]),
        "low":            float(row["low"]),
        "close":          float(row["close"]),
        "volume":         float(row["volume"]),
        "source":         row["source"],
        "fetched_at":     _parse_ts(row["fetched_at"]),
        "interval":       row["interval"],
        "is_outlier":     bool(row["is_outlier"]),
        "cleaning_notes": row.get("cleaning_notes") or None,
    }


def _parse_ts(value) -> datetime:
    """Parse a timestamp that may be a string, datetime, or pandas Timestamp."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if hasattr(value, "to_pydatetime"):
        ts = value.to_pydatetime()
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
