from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings

logger = structlog.get_logger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def init_db() -> Engine:
    """
    Create the SQLAlchemy engine and session factory.
    Call once at startup. Subsequent calls are no-ops and return the same engine.
    """
    global _engine, _SessionLocal

    if _engine is not None:
        return _engine

    if not settings.database_url:
        raise EnvironmentError(
            "Database not configured. Set DB_HOST, DB_USER, DB_PASSWORD in .env"
        )

    _engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,       # drop stale connections before use
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,        # recycle connections every 30 min (Azure PG idle timeout)
        echo=(settings.app_env == "development"),
    )

    _SessionLocal = sessionmaker(
        bind=_engine,
        autocommit=False,
        autoflush=False,
    )

    logger.info("database_initialized", host=settings.db_host, db=settings.db_name)
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy session with automatic commit/rollback.

    Usage:
        with get_session() as session:
            session.add(record)
    """
    if _SessionLocal is None:
        init_db()

    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping_db() -> bool:
    """Return True if the database is reachable, False otherwise."""
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("database_ping_failed", error=str(exc))
        return False
