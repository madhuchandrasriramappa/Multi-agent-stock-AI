"""
Ingestion Agent — orchestrates Yahoo Finance and CoinGecko clients.

Responsibilities:
- Read target symbols from config (or payload override)
- Delegate to YahooFinanceClient and CoinGeckoClient
- Return a unified result dict that downstream agents (cleaning, features) consume
- Never crash: per-source errors are captured and reported in the result

Payload keys (all optional — falls back to config defaults):
    stock_symbols  : list[str]  e.g. ["AAPL", "TSLA"]
    crypto_symbols : list[str]  e.g. ["bitcoin", "ethereum"]
    period         : str        yfinance period, default "5d"
    days           : int        CoinGecko days, default 7
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from agents.base_agent import BaseAgent
from agents.ingestion.coingecko_client import CoinGeckoClient
from agents.ingestion.yahoo_client import YahooFinanceClient
from config.settings import settings


class IngestionAgent(BaseAgent):
    """
    Fetches raw OHLCV market data for configured stocks and crypto assets.

    Output schema (AgentResult.data):
    {
        "stocks":          list[dict]   — MarketRecord dicts for equities
        "crypto":          list[dict]   — MarketRecord dicts for crypto
        "summary":         dict         — counts and succeeded/failed symbols
        "errors":          dict         — per-source error maps
        "fetched_at":      str          — ISO-8601 UTC timestamp
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._yahoo = YahooFinanceClient()
        self._coingecko = CoinGeckoClient(api_key=settings.coingecko_api_key)

    @property
    def name(self) -> str:
        return "ingestion_agent"

    def run(self, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        p = payload or {}

        stock_symbols: list[str] = p.get("stock_symbols", settings.stock_symbols_list)
        crypto_symbols: list[str] = p.get("crypto_symbols", settings.crypto_symbols_list)
        period: str = p.get("period", self._yahoo.period)
        days: int = int(p.get("days", self._coingecko.days))

        # Apply per-run overrides when provided via payload
        self._yahoo.period = period
        self._coingecko.days = days

        self.logger.info(
            "ingestion_started",
            stock_symbols=stock_symbols,
            crypto_symbols=crypto_symbols,
            period=period,
            days=days,
        )

        # ── Stocks ────────────────────────────────────────────────────────────
        stock_records, stock_errors = self._yahoo.fetch_symbols(stock_symbols)

        # ── Crypto ────────────────────────────────────────────────────────────
        crypto_records, crypto_errors = self._coingecko.fetch_coins(crypto_symbols)

        fetched_at = datetime.now(timezone.utc).isoformat()

        stock_succeeded = [s for s in stock_symbols if s not in stock_errors]
        crypto_succeeded = [c for c in crypto_symbols if c not in crypto_errors]

        self.logger.info(
            "ingestion_complete",
            stock_records=len(stock_records),
            crypto_records=len(crypto_records),
            stock_errors=len(stock_errors),
            crypto_errors=len(crypto_errors),
        )

        return {
            "stocks": [r.to_dict() for r in stock_records],
            "crypto": [r.to_dict() for r in crypto_records],
            "summary": {
                "total_records": len(stock_records) + len(crypto_records),
                "stock_records": len(stock_records),
                "crypto_records": len(crypto_records),
                "stock_symbols_succeeded": stock_succeeded,
                "stock_symbols_failed": list(stock_errors.keys()),
                "crypto_symbols_succeeded": crypto_succeeded,
                "crypto_symbols_failed": list(crypto_errors.keys()),
            },
            "errors": {
                "stocks": stock_errors,
                "crypto": crypto_errors,
            },
            "fetched_at": fetched_at,
        }
