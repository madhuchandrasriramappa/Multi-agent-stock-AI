-- Migration 002 — feature_set table
-- Phase 3: stores all computed technical indicators
-- Run via: python db/migrate.py

CREATE TABLE IF NOT EXISTS feature_set (
    id              SERIAL          PRIMARY KEY,
    symbol          VARCHAR(20)     NOT NULL,
    asset_type      VARCHAR(10)     NOT NULL,
    timestamp       TIMESTAMPTZ     NOT NULL,
    interval        VARCHAR(10)     NOT NULL,

    -- Raw reference values (avoids joins in downstream queries)
    close           NUMERIC(18, 6)  NOT NULL,
    volume          NUMERIC(24, 4)  NOT NULL,

    -- Moving averages
    sma_20          NUMERIC(18, 6),
    sma_50          NUMERIC(18, 6),
    ema_12          NUMERIC(18, 6),
    ema_26          NUMERIC(18, 6),

    -- Oscillators
    rsi_14          NUMERIC(8,  4),
    macd_line       NUMERIC(18, 6),
    macd_signal     NUMERIC(18, 6),
    macd_histogram  NUMERIC(18, 6),

    -- Bollinger Bands
    bb_upper        NUMERIC(18, 6),
    bb_middle       NUMERIC(18, 6),
    bb_lower        NUMERIC(18, 6),

    -- Volatility & VWAP
    volatility_14   NUMERIC(10, 6),
    vwap            NUMERIC(18, 6),

    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_feature_symbol_ts_interval
        UNIQUE (symbol, timestamp, interval)
);

CREATE INDEX IF NOT EXISTS idx_feature_symbol    ON feature_set (symbol);
CREATE INDEX IF NOT EXISTS idx_feature_timestamp ON feature_set (timestamp);
CREATE INDEX IF NOT EXISTS idx_feature_symbol_ts ON feature_set (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_feature_rsi       ON feature_set (symbol, rsi_14)
    WHERE rsi_14 IS NOT NULL;
