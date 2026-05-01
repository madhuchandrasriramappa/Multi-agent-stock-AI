"""
Azure Monitor / Application Insights telemetry initialisation.
Phase 0: skeleton that gracefully no-ops when not configured.
Phase 7: full OpenTelemetry trace + metric export.
"""
from __future__ import annotations

import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)


def init_telemetry() -> None:
    """
    Wire up Azure Monitor OpenTelemetry exporter.

    Safe to call even when APPLICATIONINSIGHTS_CONNECTION_STRING is not set —
    telemetry is silently skipped, which is the expected behaviour during local dev.
    """
    if not settings.applicationinsights_connection_string:
        logger.debug(
            "telemetry_skipped",
            reason="APPLICATIONINSIGHTS_CONNECTION_STRING not set",
        )
        return

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(
            connection_string=settings.applicationinsights_connection_string
        )
        logger.info("telemetry_initialized", sink="azure_monitor")

    except ImportError:
        logger.warning(
            "telemetry_unavailable",
            reason="azure-monitor-opentelemetry not installed",
        )
    except Exception as exc:
        logger.error("telemetry_init_failed", error=str(exc))
