-- Migration 003 — anomaly_alerts table
-- Phase 4: stores anomaly events flagged by the Anomaly Detection Agent
-- Run via: python db/migrate.py

CREATE TABLE IF NOT EXISTS anomaly_alerts (
    id            SERIAL          PRIMARY KEY,
    symbol        VARCHAR(20)     NOT NULL,
    asset_type    VARCHAR(10)     NOT NULL,
    timestamp     TIMESTAMPTZ     NOT NULL,
    interval      VARCHAR(10)     NOT NULL,

    -- Which detector raised the alert and on which feature
    detector      VARCHAR(30)     NOT NULL,   -- 'zscore' | 'iqr' | 'isolation_forest'
    feature       VARCHAR(50)     NOT NULL,   -- 'rsi_14' | 'volatility_14' | 'multivariate' …

    -- Raw value that triggered the alert (NULL for multivariate detectors).
    -- NUMERIC(30,4) accommodates crypto trading volumes (up to ~10^26).
    feature_value NUMERIC(30, 4),

    -- Detector-specific score: |z| for zscore, IQR-multiplier for iqr,
    -- score_samples output for isolation_forest (lower = more anomalous)
    score         NUMERIC(18, 6)  NOT NULL,

    severity      VARCHAR(10)     NOT NULL DEFAULT 'medium',  -- 'low' | 'medium' | 'high'

    created_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_anomaly_symbol_ts_interval_detector_feature
        UNIQUE (symbol, timestamp, interval, detector, feature)
);

CREATE INDEX IF NOT EXISTS idx_anomaly_symbol    ON anomaly_alerts (symbol);
CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_alerts (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_severity  ON anomaly_alerts (severity)
    WHERE severity IN ('medium', 'high');
