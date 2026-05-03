"""
AI Analyst Agent.

For every symbol in the feature_set, loads the latest technical indicators
and recent anomaly alerts, builds a structured prompt, and calls GPT-4o
(or a local mock when Azure credentials are absent) to produce a concise
market-analysis report that is persisted in analysis_reports.

Payload keys (all optional)
----------------------------
symbols        : list[str]  — limit to these symbols (default: all in feature_set)
lookback_alerts: int        — max recent alerts to include per symbol (default 20)
max_tokens     : int        — LLM max completion tokens (default 600)

Output (AgentResult.data)
--------------------------
{
  "reports_generated"   : int,
  "symbols_analyzed"    : list[str],
  "total_prompt_tokens" : int,
  "total_completion_tokens": int,
  "mode"                : "azure_openai" | "mock",
}
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from agents.analyst.llm_client import LLMClient
from agents.analyst.prompt_builder import build_system_prompt, build_user_prompt
from agents.base_agent import BaseAgent
from config.settings import settings
from db.connection import get_session
from db.models import AnalysisReport, AnomalyAlert, FeatureSet

_DEFAULT_LOOKBACK = 20
_DEFAULT_MAX_TOKENS = 600


class AnalystAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "analyst_agent"

    def run(self, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        p = payload or {}
        symbols         = p.get("symbols")
        lookback_alerts = int(p.get("lookback_alerts", _DEFAULT_LOOKBACK))
        max_tokens      = int(p.get("max_tokens", _DEFAULT_MAX_TOKENS))

        available = self._get_symbols(symbols)
        if not available:
            self.logger.warning("no_symbols_with_features")
            return {
                "reports_generated":      0,
                "symbols_analyzed":       [],
                "total_prompt_tokens":    0,
                "total_completion_tokens": 0,
                "mode": "mock",
            }

        client        = LLMClient()
        system_prompt = build_system_prompt()
        mode          = "mock" if client.is_mock else "azure_openai"

        self.logger.info("analyst_starting", symbols=available, mode=mode)

        reports_generated      = 0
        total_prompt_tokens    = 0
        total_completion_tokens = 0
        symbols_analyzed: list[str] = []

        for symbol in available:
            latest = self._load_latest_features(symbol)
            if not latest:
                self.logger.warning("no_feature_data", symbol=symbol)
                continue

            alerts     = self._load_recent_alerts(symbol, limit=lookback_alerts)
            asset_type = latest.get("asset_type", "unknown")
            user_prompt = build_user_prompt(symbol, asset_type, latest, alerts)

            self.logger.info(
                "calling_llm", symbol=symbol,
                alert_count=len(alerts), mock=client.is_mock,
            )
            text, pt, ct = client.complete(system_prompt, user_prompt, max_tokens=max_tokens)

            self._save_report(
                symbol=symbol,
                asset_type=asset_type,
                report_text=text,
                model=settings.azure_openai_deployment if not client.is_mock else "mock",
                prompt_tokens=pt,
                completion_tokens=ct,
                alert_count=len(alerts),
            )

            reports_generated       += 1
            total_prompt_tokens     += pt
            total_completion_tokens += ct
            symbols_analyzed.append(symbol)

            self.logger.info(
                "report_saved", symbol=symbol,
                mock=client.is_mock, completion_tokens=ct,
            )

        return {
            "reports_generated":       reports_generated,
            "symbols_analyzed":        symbols_analyzed,
            "total_prompt_tokens":     total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "mode":                    mode,
        }

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _get_symbols(self, symbols: list[str] | None) -> list[str]:
        if symbols:
            return sorted(symbols)
        with get_session() as session:
            rows = (
                session.query(FeatureSet.symbol)
                .filter(FeatureSet.rsi_14.isnot(None))
                .distinct()
                .all()
            )
            return sorted(r.symbol for r in rows)

    def _load_latest_features(self, symbol: str) -> dict | None:
        with get_session() as session:
            row = (
                session.query(FeatureSet)
                .filter(
                    FeatureSet.symbol == symbol,
                    FeatureSet.rsi_14.isnot(None),
                )
                .order_by(FeatureSet.timestamp.desc())
                .first()
            )
            if not row:
                return None
            return {
                "symbol":         row.symbol,
                "asset_type":     row.asset_type,
                "timestamp":      row.timestamp,
                "close":          float(row.close),
                "volume":         float(row.volume),
                "rsi_14":         float(row.rsi_14)          if row.rsi_14          else None,
                "macd_histogram": float(row.macd_histogram)  if row.macd_histogram  else None,
                "volatility_14":  float(row.volatility_14)   if row.volatility_14   else None,
                "vwap":           float(row.vwap)            if row.vwap            else None,
                "sma_20":         float(row.sma_20)          if row.sma_20          else None,
                "ema_12":         float(row.ema_12)          if row.ema_12          else None,
            }

    def _load_recent_alerts(self, symbol: str, limit: int = _DEFAULT_LOOKBACK) -> list[dict]:
        with get_session() as session:
            rows = (
                session.query(AnomalyAlert)
                .filter(AnomalyAlert.symbol == symbol)
                .order_by(AnomalyAlert.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [{
                "timestamp":     r.timestamp,
                "detector":      r.detector,
                "feature":       r.feature,
                "feature_value": float(r.feature_value) if r.feature_value else None,
                "score":         float(r.score),
                "severity":      r.severity,
            } for r in rows]

    def _save_report(
        self,
        symbol: str,
        asset_type: str,
        report_text: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        alert_count: int,
    ) -> None:
        with get_session() as session:
            session.add(AnalysisReport(
                symbol=symbol,
                asset_type=asset_type,
                generated_at=datetime.now(timezone.utc),
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                report_text=report_text,
                alert_count=alert_count,
            ))
