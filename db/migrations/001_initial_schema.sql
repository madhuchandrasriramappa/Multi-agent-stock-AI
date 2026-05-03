-- Migration 001 — Initial schema
-- Phase 2: raw_market_data + clean_market_data
-- Run via: python db/migrate.py
-- Or manually: psql -U stockai_user -d stockai -f db/migrations/001_initial_schema.sql

-- ── Raw market data ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw_market_data (
    id          SERIAL          PRIMARY KEY,
    symbol      VARCHAR(20)     NOT NULL,
    asset_type  VARCHAR(10)     NOT NULL,
    timestamp   TIMESTAMPTZ     NOT NULL,
    open        NUMERIC(18, 6)  NOT NULL,
    high        NUMERIC(18, 6)  NOT NULL,
    low         NUMERIC(18, 6)  NOT NULL,
    close       NUMERIC(18, 6)  NOT NULL,
    volume      NUMERIC(24, 4)  NOT NULL DEFAULT 0,
    source      VARCHAR(30)     NOT NULL,
    fetched_at  TIMESTAMPTZ     NOT NULL,
    interval    VARCHAR(10)     NOT NULL,
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_raw_symbol_ts_source_interval
        UNIQUE (symbol, timestamp, source, interval)
);

CREATE INDEX IF NOT EXISTS idx_raw_symbol    ON raw_market_data (symbol);
CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_market_data (timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_symbol_ts ON raw_market_data (symbol, timestamp DESC);

-- ── Clean market data ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS clean_market_data (
    id              SERIAL          PRIMARY KEY,
    raw_id          INTEGER         REFERENCES raw_market_data(id) ON DELETE SET NULL,
    symbol          VARCHAR(20)     NOT NULL,
    asset_type      VARCHAR(10)     NOT NULL,
    timestamp       TIMESTAMPTZ     NOT NULL,
    open            NUMERIC(18, 6)  NOT NULL,
    high            NUMERIC(18, 6)  NOT NULL,
    low             NUMERIC(18, 6)  NOT NULL,
    close           NUMERIC(18, 6)  NOT NULL,
    volume          NUMERIC(24, 4)  NOT NULL DEFAULT 0,
    source          VARCHAR(30)     NOT NULL,
    fetched_at      TIMESTAMPTZ     NOT NULL,
    interval        VARCHAR(10)     NOT NULL,
    is_outlier      BOOLEAN         NOT NULL DEFAULT FALSE,
    cleaning_notes  TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_clean_symbol_ts_source_interval
        UNIQUE (symbol, timestamp, source, interval)
);

CREATE INDEX IF NOT EXISTS idx_clean_symbol    ON clean_market_data (symbol);
CREATE INDEX IF NOT EXISTS idx_clean_timestamp ON clean_market_data (timestamp);
CREATE INDEX IF NOT EXISTS idx_clean_symbol_ts ON clean_market_data (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_clean_outlier   ON clean_market_data (symbol, is_outlier)
    WHERE is_outlier = TRUE;
