"""
CoinGecko REST API client — returns normalised MarketRecord objects.

Endpoint strategy
-----------------
/coins/{id}/ohlc?vs_currency=usd&days=7
    Returns [[ts_ms, open, high, low, close], ...] at 4-hour candles.

/coins/{id}/market_chart?vs_currency=usd&days=7
    Returns {"prices": [...], "total_volumes": [[ts_ms, vol], ...]}
    at hourly granularity for ≤ 90 days.

The two responses are merged: OHLC from the first endpoint, volume
aggregated from the second with pandas resample("4h").sum().

Rate limiting: CoinGecko free tier allows ~10-15 req/min on the public
endpoint.  Each fetch_coin() call costs 2 requests; we cap at 6 coins/min.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
import structlog

from agents.ingestion.rate_limiter import RateLimiter
from agents.ingestion.retry import with_retry
from agents.ingestion.schemas import MarketRecord

logger = structlog.get_logger(client="coingecko")

_BASE_URL = "https://api.coingecko.com/api/v3"
_DEFAULT_DAYS = 7          # 7 days → 4-hour OHLC candles
_RATE_LIMIT_RPM = 12       # conservative: 1 call per 5 s


class CoinGeckoClient:
    """Fetches OHLCV data for crypto assets via the CoinGecko public API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        days: int = _DEFAULT_DAYS,
    ) -> None:
        self.days = days
        self._rate_limiter = RateLimiter(calls_per_minute=_RATE_LIMIT_RPM)
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        if api_key:
            # CoinGecko demo key (free with registration) raises the limit to 30 rpm
            self._session.headers.update({"x-cg-demo-api-key": api_key})
        self.logger = structlog.get_logger(client="coingecko")

    # ── Private helpers ────────────────────────────────────────────────────────

    @with_retry(max_attempts=3, base_delay=5.0, backoff_factor=2.0)
    def _get_ohlc(self, coin_id: str) -> list[list]:
        """Fetch raw OHLC array: [[ts_ms, open, high, low, close], ...]"""
        self._rate_limiter.acquire()
        url = f"{_BASE_URL}/coins/{coin_id}/ohlc"
        resp = self._session.get(
            url,
            params={"vs_currency": "usd", "days": self.days},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self.logger.debug("ohlc_raw_rows", coin=coin_id, rows=len(data))
        return data

    @with_retry(max_attempts=3, base_delay=5.0, backoff_factor=2.0)
    def _get_market_chart(self, coin_id: str) -> dict:
        """Fetch market chart with total_volumes timeseries."""
        self._rate_limiter.acquire()
        url = f"{_BASE_URL}/coins/{coin_id}/market_chart"
        resp = self._session.get(
            url,
            params={"vs_currency": "usd", "days": self.days},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def _merge_ohlc_with_volume(
        self,
        coin_id: str,
        ohlc_raw: list[list],
        volumes_raw: list[list],
        fetched_at: datetime,
    ) -> list[MarketRecord]:
        """
        Merge OHLC and volume arrays into MarketRecord objects.

        OHLC comes in at 4-hour candles; volumes from market_chart come in
        at hourly intervals for the same period.  We resample volumes to 4h
        by summing, then join on timestamp.
        """
        if not ohlc_raw:
            return []

        # --- OHLC -------------------------------------------------------
        ohlc_df = pd.DataFrame(
            ohlc_raw, columns=["ts_ms", "open", "high", "low", "close"]
        )
        ohlc_df["timestamp"] = pd.to_datetime(ohlc_df["ts_ms"], unit="ms", utc=True)
        ohlc_df = (
            ohlc_df.set_index("timestamp")
            .drop(columns=["ts_ms"])
            .sort_index()
        )

        # --- Volume (hourly → 4 h) ----------------------------------------
        interval_str = "4h"
        if volumes_raw:
            vol_df = pd.DataFrame(volumes_raw, columns=["ts_ms", "volume"])
            vol_df["timestamp"] = pd.to_datetime(vol_df["ts_ms"], unit="ms", utc=True)
            vol_df = vol_df.set_index("timestamp").sort_index()
            vol_resampled = vol_df["volume"].resample(interval_str).sum()
            merged = ohlc_df.join(vol_resampled, how="left")
            merged["volume"] = merged["volume"].fillna(0.0)
        else:
            merged = ohlc_df.copy()
            merged["volume"] = 0.0

        merged = merged.dropna(subset=["open", "high", "low", "close"])

        records: list[MarketRecord] = []
        for ts, row in merged.iterrows():
            records.append(
                MarketRecord(
                    symbol=coin_id,
                    asset_type="crypto",
                    timestamp=ts.to_pydatetime(),
                    open=round(float(row["open"]), 4),
                    high=round(float(row["high"]), 4),
                    low=round(float(row["low"]), 4),
                    close=round(float(row["close"]), 4),
                    volume=round(float(row["volume"]), 2),
                    source="coingecko",
                    fetched_at=fetched_at,
                    interval=interval_str,
                )
            )
        return records

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch_coin(self, coin_id: str) -> list[MarketRecord]:
        """
        Fetch OHLCV records for a single CoinGecko coin ID (e.g. "bitcoin").

        Makes 2 API calls (OHLC + market chart), merges them, and returns
        a list of MarketRecord objects.  Returns [] if the coin returns no data.
        """
        self.logger.info("fetching_coin", coin=coin_id, days=self.days)
        fetched_at = datetime.now(timezone.utc)

        ohlc_raw = self._get_ohlc(coin_id)
        chart = self._get_market_chart(coin_id)
        volumes_raw: list[list] = chart.get("total_volumes", [])

        records = self._merge_ohlc_with_volume(coin_id, ohlc_raw, volumes_raw, fetched_at)
        self.logger.info("coin_fetched", coin=coin_id, records=len(records))
        return records

    def fetch_coins(
        self, coin_ids: list[str]
    ) -> tuple[list[MarketRecord], dict[str, str]]:
        """
        Fetch multiple coins.  Per-coin failures are collected; they never
        prevent other coins from being fetched.

        Returns:
            (all_records, errors)  where errors maps coin_id → error message
        """
        all_records: list[MarketRecord] = []
        errors: dict[str, str] = {}

        for coin_id in coin_ids:
            try:
                records = self.fetch_coin(coin_id)
                all_records.extend(records)
            except Exception as exc:
                self.logger.error(
                    "coin_failed",
                    coin=coin_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                errors[coin_id] = str(exc)

        self.logger.info(
            "batch_complete",
            total=len(all_records),
            succeeded=len(coin_ids) - len(errors),
            failed=len(errors),
        )
        return all_records, errors
