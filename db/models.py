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


class FeatureSet(Base):
    """
    Computed technical indicators for every (symbol, timestamp, interval) row
    in clean_market_data.  Produced by the Feature Engineering Agent.

    All indicator columns are nullable — early rows in a series don't have
    enough history to compute a 20-period moving average, for example.
    Phase 4 (Anomaly Detection) only reads rows where the needed columns
    are non-null.
    """
    __tablename__ = "feature_set"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "timestamp", "interval",
            name="uq_feature_symbol_ts_interval",
        ),
    )

    id               = Column(Integer, primary_key=True, autoincrement=True)
    symbol           = Column(String(20),     nullable=False, index=True)
    asset_type       = Column(String(10),     nullable=False)
    timestamp        = Column(DateTime(timezone=True), nullable=False, index=True)
    interval         = Column(String(10),     nullable=False)

    # Raw price/volume kept for reference (avoids joins in downstream queries)
    close            = Column(Numeric(18, 6), nullable=False)
    volume           = Column(Numeric(24, 4), nullable=False)

    # ── Moving averages ────────────────────────────────────────────────────────
    sma_20           = Column(Numeric(18, 6), nullable=True)
    sma_50           = Column(Numeric(18, 6), nullable=True)
    ema_12           = Column(Numeric(18, 6), nullable=True)
    ema_26           = Column(Numeric(18, 6), nullable=True)

    # ── Oscillators ───────────────────────────────────────────────────────────
    rsi_14           = Column(Numeric(8,  4),  nullable=True)   # 0 – 100
    macd_line        = Column(Numeric(18, 6),  nullable=True)
    macd_signal      = Column(Numeric(18, 6),  nullable=True)
    macd_histogram   = Column(Numeric(18, 6),  nullable=True)

    # ── Bollinger Bands (20-period, 2 std devs) ───────────────────────────────
    bb_upper         = Column(Numeric(18, 6), nullable=True)
    bb_middle        = Column(Numeric(18, 6), nullable=True)
    bb_lower         = Column(Numeric(18, 6), nullable=True)

    # ── Volatility & VWAP ────────────────────────────────────────────────────
    volatility_14    = Column(Numeric(10, 6), nullable=True)    # rolling std of returns
    vwap             = Column(Numeric(18, 6), nullable=True)    # resets each UTC day

    created_at       = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self) -> str:
        return f"<FeatureSet {self.symbol} {self.timestamp} rsi={self.rsi_14}>"


class AnomalyAlert(Base):
    """
    One anomaly event raised by the Anomaly Detection Agent.

    detector   : which algorithm fired ('zscore', 'iqr', 'isolation_forest')
    feature    : which indicator was anomalous ('rsi_14', 'multivariate', …)
    score      : detector-specific magnitude (|z|, IQR multiplier, IF score)
    severity   : 'low' | 'medium' | 'high' derived from score thresholds
    """
    __tablename__ = "anomaly_alerts"
    __table_args__ = (
        UniqueConstraint(
            "symbol", "timestamp", "interval", "detector", "feature",
            name="uq_anomaly_symbol_ts_interval_detector_feature",
        ),
    )

    id            = Column(Integer, primary_key=True, autoincrement=True)
    symbol        = Column(String(20),     nullable=False, index=True)
    asset_type    = Column(String(10),     nullable=False)
    timestamp     = Column(DateTime(timezone=True), nullable=False, index=True)
    interval      = Column(String(10),     nullable=False)
    detector      = Column(String(30),     nullable=False)
    feature       = Column(String(50),     nullable=False)
    feature_value = Column(Numeric(30, 4), nullable=True)
    score         = Column(Numeric(18, 6), nullable=False)
    severity      = Column(String(10),     nullable=False, default="medium")
    created_at    = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self) -> str:
        return (
            f"<AnomalyAlert {self.symbol} {self.timestamp} "
            f"{self.detector}/{self.feature} score={self.score} [{self.severity}]>"
        )


class AnalysisReport(Base):
    """
    One GPT-4o generated market analysis for a single symbol.

    A new row is inserted on every pipeline run — full history is kept.
    model = 'mock' when Azure OpenAI is not configured (local dev).
    """
    __tablename__ = "analysis_reports"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    symbol            = Column(String(20), nullable=False, index=True)
    asset_type        = Column(String(10), nullable=False)
    generated_at      = Column(DateTime(timezone=True), nullable=False)
    model             = Column(String(50), nullable=False)
    prompt_tokens     = Column(Integer,    nullable=False, default=0)
    completion_tokens = Column(Integer,    nullable=False, default=0)
    report_text       = Column(Text,       nullable=False)
    alert_count       = Column(Integer,    nullable=False, default=0)
    created_at        = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self) -> str:
        return f"<AnalysisReport {self.symbol} {self.generated_at} model={self.model}>"
