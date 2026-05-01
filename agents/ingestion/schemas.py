"""
Shared data schema for the ingestion layer.

MarketRecord is the normalised OHLCV envelope produced by every data-source
client.  All downstream agents (cleaning, features, anomaly) consume this
schema — changing it here changes the contract for the whole pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

AssetType = Literal["stock", "crypto"]
DataSource = Literal["yahoo_finance", "coingecko"]


@dataclass
class MarketRecord:
    """
    One OHLCV candle from any supported data source.

    Fields
    ------
    symbol     : ticker ("AAPL") or CoinGecko coin ID ("bitcoin")
    asset_type : "stock" | "crypto"
    timestamp  : candle open time, always UTC-aware
    open / high / low / close : prices in USD, rounded to 4 d.p.
    volume     : traded volume in native units (shares or coin units)
    source     : which provider produced this record
    fetched_at : wall-clock UTC time the record was retrieved
    interval   : candle width string — "1h", "4h", "30m", "1d"
    """

    symbol: str
    asset_type: AssetType
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: DataSource
    fetched_at: datetime
    interval: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "asset_type": self.asset_type,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "interval": self.interval,
        }
