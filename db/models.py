"""
SQLAlchemy ORM models — single source of truth for all database tables.
Phase 0: Base only. Table models are added in Phase 2 (storage sprint).
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base.  All model classes inherit from this."""
    pass


# ── Table models added in Phase 2 ─────────────────────────────────────────────
# RawMarketData    — raw tick/OHLCV rows from ingestion
# CleanMarketData  — validated, de-duped rows from cleaning agent
# FeatureSet       — indicators computed by feature engineering agent
# Anomaly          — flagged events from anomaly detection agent
