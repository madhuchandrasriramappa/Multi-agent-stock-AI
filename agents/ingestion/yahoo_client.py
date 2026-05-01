"""
Yahoo Finance client — wraps yfinance to return normalised MarketRecord objects.

Design decisions:
- Fetches one symbol at a time for granular error handling (bulk download
  silently drops failed symbols which is harder to detect).
- auto_adjust=True so splits and dividends are baked into prices.
- NaN rows are dropped before conversion; an empty result is not an error.
- Retry is applied per-symbol; a 3-strike failure logs and moves on.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
import yfinance as yf

from agents.ingestion.retry import with_retry
from agents.ingestion.schemas import MarketRecord

logger = structlog.get_logger(client="yahoo_finance")

# Default fetch window: 5 trading days at 1-hour granularity.
# 15m is more granular but unreliable outside US market hours.
_DEFAULT_PERIOD = "5d"
_DEFAULT_INTERVAL = "1h"


class YahooFinanceClient:
    """Fetches OHLCV data for US-listed equities via yfinance."""

    def __init__(
        self,
        period: str = _DEFAULT_PERIOD,
        interval: str = _DEFAULT_INTERVAL,
    ) -> None:
        self.period = period
        self.interval = interval
        self.logger = structlog.get_logger(client="yahoo_finance")

    @with_retry(max_attempts=3, base_delay=2.0, backoff_factor=2.0)
    def fetch_symbol(self, symbol: str) -> list[MarketRecord]:
        """
        Fetch OHLCV history for a single ticker.

        Returns an empty list when the symbol yields no data (e.g. market
        closed, bad ticker) rather than raising — callers decide what to do.
        """
        self.logger.info(
            "fetching_symbol",
            symbol=symbol,
            period=self.period,
            interval=self.interval,
        )

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            period=self.period,
            interval=self.interval,
            auto_adjust=True,
            prepost=False,   # exclude pre/post-market — noisy for our use case
        )

        if df.empty:
            self.logger.warning("empty_response", symbol=symbol)
            return []

        # Drop rows where any price column is NaN (common at market boundaries)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

        fetched_at = datetime.now(timezone.utc)
        records: list[MarketRecord] = []

        for ts, row in df.iterrows():
            # yfinance intraday timestamps are timezone-aware (US/Eastern or UTC)
            timestamp = ts.to_pydatetime()
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)

            records.append(
                MarketRecord(
                    symbol=symbol.upper(),
                    asset_type="stock",
                    timestamp=timestamp,
                    open=round(float(row["Open"]), 4),
                    high=round(float(row["High"]), 4),
                    low=round(float(row["Low"]), 4),
                    close=round(float(row["Close"]), 4),
                    volume=float(row["Volume"]),
                    source="yahoo_finance",
                    fetched_at=fetched_at,
                    interval=self.interval,
                )
            )

        self.logger.info("symbol_fetched", symbol=symbol, records=len(records))
        return records

    def fetch_symbols(
        self,
        symbols: list[str],
    ) -> tuple[list[MarketRecord], dict[str, str]]:
        """
        Fetch a list of symbols.  Per-symbol failures are collected into
        `errors` and do not prevent other symbols from being fetched.

        Returns:
            (all_records, errors)  where errors maps symbol → error message
        """
        all_records: list[MarketRecord] = []
        errors: dict[str, str] = {}

        for symbol in symbols:
            try:
                records = self.fetch_symbol(symbol)
                all_records.extend(records)
            except Exception as exc:
                self.logger.error(
                    "symbol_failed",
                    symbol=symbol,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                errors[symbol] = str(exc)

        self.logger.info(
            "batch_complete",
            total=len(all_records),
            succeeded=len(symbols) - len(errors),
            failed=len(errors),
        )
        return all_records, errors
