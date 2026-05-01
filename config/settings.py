from __future__ import annotations

from typing import Optional
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── Azure OpenAI ───────────────────────────────────────────────────────────
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-02-15-preview"

    # ── Azure PostgreSQL ───────────────────────────────────────────────────────
    db_host: Optional[str] = None
    db_port: int = 5432
    db_name: str = "stockai"
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_ssl_mode: str = "require"

    # ── Azure Service Bus ──────────────────────────────────────────────────────
    azure_servicebus_connection_string: Optional[str] = None
    azure_servicebus_queue_name: str = "pipeline-events"

    # ── Azure Key Vault ────────────────────────────────────────────────────────
    azure_keyvault_url: Optional[str] = None

    # ── Azure Monitor / App Insights ───────────────────────────────────────────
    applicationinsights_connection_string: Optional[str] = None

    # ── External APIs ──────────────────────────────────────────────────────────
    coingecko_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None

    # ── Ingestion config ───────────────────────────────────────────────────────
    stock_symbols: str = "AAPL,MSFT,GOOGL,TSLA,NVDA"
    crypto_symbols: str = "bitcoin,ethereum,solana,cardano"
    ingestion_interval_minutes: int = 15

    # ── Computed fields ────────────────────────────────────────────────────────

    @computed_field
    @property
    def database_url(self) -> Optional[str]:
        if not all([self.db_host, self.db_user, self.db_password]):
            return None
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?sslmode={self.db_ssl_mode}"
        )

    @computed_field
    @property
    def stock_symbols_list(self) -> list[str]:
        return [s.strip() for s in self.stock_symbols.split(",") if s.strip()]

    @computed_field
    @property
    def crypto_symbols_list(self) -> list[str]:
        return [s.strip() for s in self.crypto_symbols.split(",") if s.strip()]

    def require(self, *field_names: str) -> None:
        """Assert that the given config fields are set. Call in agent __init__ before use."""
        missing = [f for f in field_names if not getattr(self, f, None)]
        if missing:
            raise EnvironmentError(
                f"Missing required config: {', '.join(f.upper() for f in missing)}. "
                f"Check your .env file against .env.example."
            )


settings = Settings()
