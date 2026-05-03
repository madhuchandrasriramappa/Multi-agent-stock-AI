#!/usr/bin/env python3
"""
Database migration runner.

Uses SQLAlchemy's create_all() which is idempotent —
safe to run repeatedly; it only creates tables that don't exist yet.

Usage:
    python db/migrate.py
    # or via CLI:
    python main.py migrate
"""
from __future__ import annotations

import structlog

from config.logging_config import configure_logging
from config.settings import settings
from db.connection import init_db
from db.models import Base

logger = structlog.get_logger("migrate")


def run_migrations() -> None:
    configure_logging(log_level=settings.log_level, app_env=settings.app_env)

    if not settings.database_url:
        raise EnvironmentError(
            "Database not configured.\n"
            "For local dev, start Docker Postgres:\n"
            "  docker-compose up -d\n"
            "Then set in .env:\n"
            "  DB_HOST=localhost\n"
            "  DB_USER=stockai_user\n"
            "  DB_PASSWORD=localdev123\n"
            "  DB_SSL_MODE=disable"
        )

    engine = init_db()
    logger.info("running_migrations", host=settings.db_host, db=settings.db_name)

    Base.metadata.create_all(engine)

    tables = list(Base.metadata.tables.keys())
    for table in tables:
        logger.info("table_ready", table=table)

    logger.info("migrations_complete", tables=tables)


if __name__ == "__main__":
    run_migrations()
