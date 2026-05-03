-- Migration 004 — analysis_reports table
-- Phase 5: stores GPT-4o generated market analysis reports
-- Run via: python db/migrate.py

CREATE TABLE IF NOT EXISTS analysis_reports (
    id                SERIAL          PRIMARY KEY,
    symbol            VARCHAR(20)     NOT NULL,
    asset_type        VARCHAR(10)     NOT NULL,
    generated_at      TIMESTAMPTZ     NOT NULL,

    -- LLM metadata
    model             VARCHAR(50)     NOT NULL,   -- e.g. 'gpt-4o', 'mock'
    prompt_tokens     INT             NOT NULL DEFAULT 0,
    completion_tokens INT             NOT NULL DEFAULT 0,

    -- The actual report
    report_text       TEXT            NOT NULL,
    alert_count       INT             NOT NULL DEFAULT 0,

    created_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
    -- No unique constraint: each pipeline run appends a new report (full history)
);

CREATE INDEX IF NOT EXISTS idx_reports_symbol    ON analysis_reports (symbol);
CREATE INDEX IF NOT EXISTS idx_reports_generated ON analysis_reports (generated_at DESC);
