"""
SQLAlchemy ORM models — single source of truth for all database tables.

Tables
------
raw_market_data   : every record exactly as received from the data source
clean_market_data : validated, de-duped records produced by the Cleaning Agent
                    (Feature Engineering and Anomaly Detection read from here)
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Shared declarative base — all ORM models inherit from this."""
    pass


class RawMarketData(Base):
    """
    One OHLCV candle exactly as received from Yahoo Finance or CoinGecko.
    Nothing is modified after ingestion; this table is append-only.

    Unique constraint on (symbol, timestamp, source, interval) prevents
    duplicate ingestion runs from creating duplicate rows.
    """
    __tablename__ = "raw_market_data"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "timestamp", "source", "interval",
            name="uq_raw_symbol_ts_source_interval",
        ),
    )

    id         = Column(Integer, primary_key=True, autoincrement=True)
    symbol     = Column(String(20),    nullable=False, index=True)
    asset_type = Column(String(10),    nullable=False)
    timestamp  = Column(DateTime(timezone=True), nullable=False, index=True)
    open       = Column(Numeric(18, 6), nullable=False)
    high       = Column(Numeric(18, 6), nullable=False)
    low        = Column(Numeric(18, 6), nullable=False)
    close      = Column(Numeric(18, 6), nullable=False)
    volume     = Column(Numeric(24, 4), nullable=False, default=0)
    source     = Column(String(30),    nullable=False)
    fetched_at = Column(DateTime(timezone=True), nullable=False)
    interval   = Column(String(10),    nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self) -> str:
        return f"<RawMarketData {self.symbol} {self.timestamp} close={self.close}>"


class CleanMarketData(Base):
    """
    Validated, de-duplicated OHLCV record produced by the Cleaning Agent.

    raw_id traces back to the source row in raw_market_data.
    is_outlier flags extreme single-candle price moves (>20%) for
    downstream attention — the row is kept, not deleted.
    cleaning_notes records what rule triggered the outlier flag.
    """
    __tablename__ = "clean_market_data"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "timestamp", "source", "interval",
            name="uq_clean_symbol_ts_source_interval",
        ),
    )

    id             = Column(Integer, primary_key=True, autoincrement=True)
    raw_id         = Column(Integer, ForeignKey("raw_market_data.id", ondelete="SET NULL"), nullable=True)
    symbol         = Column(String(20),    nullable=False, index=True)
    asset_type     = Column(String(10),    nullable=False)
    timestamp      = Column(DateTime(timezone=True), nullable=False, index=True)
    open           = Column(Numeric(18, 6), nullable=False)
    high           = Column(Numeric(18, 6), nullable=False)
    low            = Column(Numeric(18, 6), nullable=False)
    close          = Column(Numeric(18, 6), nullable=False)
    volume         = Column(Numeric(24, 4), nullable=False, default=0)
    source         = Column(String(30),    nullable=False)
    fetched_at     = Column(DateTime(timezone=True), nullable=False)
    interval       = Column(String(10),    nullable=False)
    is_outlier     = Column(Boolean,       nullable=False, default=False)
    cleaning_notes = Column(Text,          nullable=True)
    created_at     = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self) -> str:
        flag = " [OUTLIER]" if self.is_outlier else ""
        return f"<CleanMarketData {self.symbol} {self.timestamp} close={self.close}{flag}>"


# ── Added in later phases ──────────────────────────────────────────────────────
# FeatureSet  — Phase 3: moving averages, RSI, Bollinger Bands, volatility
# Anomaly     — Phase 4: Z-score / IQR anomaly events
