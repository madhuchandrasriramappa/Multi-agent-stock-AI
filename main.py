#!/usr/bin/env python3
"""
Multi-Agent Stock & Crypto AI Intelligence System — CLI entry point.

Commands:
  run     Run the full pipeline or a specific agent
  query   Ask the AI Analyst a natural-language question
  status  Check connectivity to all configured services
"""
from __future__ import annotations

import click

from config.logging_config import configure_logging
from config.settings import settings
from monitoring.telemetry import init_telemetry


def _bootstrap() -> None:
    configure_logging(log_level=settings.log_level, app_env=settings.app_env)
    init_telemetry()


# ── CLI group ──────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="stock-ai")
def cli() -> None:
    """Multi-Agent Stock & Crypto AI Intelligence System."""
    _bootstrap()


# ── Commands ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--agent",
    type=click.Choice(
        ["ingestion", "cleaning", "features", "anomaly", "analyst", "all"],
        case_sensitive=False,
    ),
    default="all",
    show_default=True,
    help="Which agent to run.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate config and agent wiring without executing any work.",
)
def run(agent: str, dry_run: bool) -> None:
    """Run the full pipeline or a specific agent."""
    if dry_run:
        click.echo(f"[dry-run] Would run agent: {agent}")
        click.echo("[dry-run] Config loaded successfully — no agents executed.")
        return

    click.echo(
        f"[Phase 0] Pipeline runner wires up in Phase 6 (Orchestrator). "
        f"Agent targeted: {agent}"
    )


@cli.command()
@click.argument("question")
@click.option(
    "--symbol",
    default=None,
    metavar="SYMBOL",
    help="Narrow context to a specific ticker or coin, e.g. AAPL or bitcoin.",
)
def query(question: str, symbol: str | None) -> None:
    """Ask the AI Analyst a natural-language question about market data."""
    click.echo("[Phase 0] AI Analyst agent arrives in Phase 5.")
    click.echo(f"  Question : {question}")
    click.echo(f"  Symbol   : {symbol or 'all'}")


@cli.command()
def status() -> None:
    """Check connectivity to all configured services."""
    from db.connection import ping_db

    rows: list[tuple[str, bool, str]] = [
        ("Azure OpenAI",    bool(settings.azure_openai_endpoint and settings.azure_openai_api_key), ""),
        ("Azure Service Bus", bool(settings.azure_servicebus_connection_string), ""),
        ("Azure Key Vault",   bool(settings.azure_keyvault_url), ""),
        ("App Insights",      bool(settings.applicationinsights_connection_string), ""),
        ("CoinGecko API",     True, "free tier — no key required"),
    ]

    # Live DB ping only when credentials exist
    if settings.database_url:
        reachable = ping_db()
        rows.insert(0, ("PostgreSQL", reachable, "live ping" if reachable else "unreachable"))
    else:
        rows.insert(0, ("PostgreSQL", False, "not configured"))

    click.echo()
    click.echo("  System Status")
    click.echo("  " + "─" * 55)
    for name, ok, note in rows:
        icon  = click.style("✓", fg="green") if ok else click.style("✗", fg="red")
        state = "configured" if ok else "not configured"
        suffix = f"  ({note})" if note else ""
        click.echo(f"  {icon}  {name:<28} {state}{suffix}")
    click.echo()


if __name__ == "__main__":
    cli()
