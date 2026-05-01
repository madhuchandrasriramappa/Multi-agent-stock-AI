from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog


@dataclass
class AgentResult:
    """Typed return value from every agent's execute() call."""

    status: str                          # "success" | "error"
    agent: str
    elapsed_seconds: float
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    def __repr__(self) -> str:
        if self.succeeded:
            keys = list(self.data.keys()) if self.data else []
            return f"AgentResult(agent={self.agent!r}, status=success, data_keys={keys}, elapsed={self.elapsed_seconds}s)"
        return f"AgentResult(agent={self.agent!r}, status=error, error={self.error!r})"


class BaseAgent(ABC):
    """
    Abstract base class for every agent in the pipeline.

    Subclasses must implement:
      - name (property) — unique string identifier, e.g. "ingestion_agent"
      - run(payload)    — core logic; return a plain dict of results

    Call execute() from orchestration code, not run() directly.
    execute() adds timing, structured logging, and error handling.
    """

    def __init__(self) -> None:
        self.logger = structlog.get_logger(agent=self.name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier used in logs and AgentResult."""
        ...

    @abstractmethod
    def run(self, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Core agent logic.

        Args:
            payload: optional data passed in from the orchestrator or upstream agent.

        Returns:
            A plain dict whose contents are agent-specific. Returned as
            AgentResult.data when the call succeeds.
        """
        ...

    def execute(self, payload: Optional[dict[str, Any]] = None) -> AgentResult:
        """
        Public entry point for all orchestration code.

        Wraps run() with:
          - structured start/complete/error log events
          - wall-clock timing
          - exception capture (agents never crash the orchestrator)
        """
        self.logger.info(
            "agent_started",
            payload_keys=list(payload.keys()) if payload else [],
        )
        start = time.perf_counter()

        try:
            data = self.run(payload)
            elapsed = round(time.perf_counter() - start, 3)
            self.logger.info(
                "agent_completed",
                elapsed_seconds=elapsed,
                result_keys=list(data.keys()) if data else [],
            )
            return AgentResult(
                status="success",
                agent=self.name,
                elapsed_seconds=elapsed,
                data=data,
            )

        except Exception as exc:
            elapsed = round(time.perf_counter() - start, 3)
            self.logger.error(
                "agent_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                elapsed_seconds=elapsed,
                exc_info=True,
            )
            return AgentResult(
                status="error",
                agent=self.name,
                elapsed_seconds=elapsed,
                error=str(exc),
                error_type=type(exc).__name__,
            )
