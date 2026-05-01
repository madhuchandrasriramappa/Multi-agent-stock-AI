"""
Phase 0 tests — verifies scaffold wiring without any Azure credentials.
Run:  pytest tests/test_phase0.py -v
"""
import pytest


# ── Settings ───────────────────────────────────────────────────────────────────

def test_settings_import():
    from config.settings import settings
    assert settings is not None


def test_default_values():
    from config.settings import settings
    assert settings.app_env in ("development", "staging", "production")
    assert settings.log_level.upper() in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    assert settings.db_port == 5432
    assert settings.ingestion_interval_minutes > 0


def test_stock_symbol_parsing():
    from config.settings import settings
    symbols = settings.stock_symbols_list
    assert isinstance(symbols, list)
    assert len(symbols) > 0
    assert all(isinstance(s, str) and len(s) > 0 for s in symbols)


def test_crypto_symbol_parsing():
    from config.settings import settings
    symbols = settings.crypto_symbols_list
    assert isinstance(symbols, list)
    assert len(symbols) > 0


def test_database_url_none_without_db_config(monkeypatch):
    """database_url should be None when DB credentials are missing — no crash."""
    from config import settings as settings_module
    from config.settings import Settings

    # Construct a fresh Settings with no DB env vars
    fresh = Settings(
        db_host=None,
        db_user=None,
        db_password=None,
    )
    assert fresh.database_url is None


def test_require_raises_on_missing_field():
    from config.settings import Settings
    s = Settings(azure_openai_endpoint=None)
    with pytest.raises(EnvironmentError, match="AZURE_OPENAI_ENDPOINT"):
        s.require("azure_openai_endpoint")


def test_require_passes_when_field_set():
    from config.settings import Settings
    s = Settings(azure_openai_endpoint="https://test.openai.azure.com/")
    s.require("azure_openai_endpoint")  # must not raise


# ── Base Agent ─────────────────────────────────────────────────────────────────

def test_base_agent_contract():
    """Concrete subclass correctly calls run() and returns AgentResult."""
    from agents.base_agent import BaseAgent, AgentResult

    class EchoAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "echo_agent"

        def run(self, payload=None):
            return {"echoed": payload}

    agent = EchoAgent()
    result = agent.execute({"msg": "hello"})

    assert isinstance(result, AgentResult)
    assert result.succeeded
    assert result.agent == "echo_agent"
    assert result.data == {"echoed": {"msg": "hello"}}
    assert result.elapsed_seconds >= 0


def test_base_agent_captures_exception():
    """execute() must not propagate exceptions — they're captured in AgentResult."""
    from agents.base_agent import BaseAgent

    class BrokenAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "broken_agent"

        def run(self, payload=None):
            raise ValueError("simulated failure")

    result = BrokenAgent().execute()

    assert not result.succeeded
    assert result.status == "error"
    assert result.error == "simulated failure"
    assert result.error_type == "ValueError"


# ── Logging config ─────────────────────────────────────────────────────────────

def test_logging_configure_does_not_raise():
    from config.logging_config import configure_logging
    configure_logging(log_level="DEBUG", app_env="development")  # must not raise


# ── Messaging skeleton ─────────────────────────────────────────────────────────

def test_pipeline_event_serialisation():
    from messaging.service_bus import PipelineEvent
    event = PipelineEvent(event_type="ingestion_complete", payload={"symbols": ["AAPL"]})
    raw = event.to_json()
    restored = PipelineEvent.from_json(raw)
    assert restored.event_type == "ingestion_complete"
    assert restored.payload["symbols"] == ["AAPL"]
