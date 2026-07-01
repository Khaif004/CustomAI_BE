"""Async SQLAlchemy engine + session management for the Tool Registry.

The rest of the backend talks to Neon over sync psycopg2 (see
``app/api/apps.py:_neon_conn`` and ``app/knowledge/vector_store.py``). This
module adds a *separate*, async engine (psycopg v3) used ONLY by the
tool-catalog feature. The two pools never share connections.

Design choices for the Neon pgbouncer **pooler** endpoint:
  * Driver = psycopg v3 (``postgresql+psycopg://``) so the libpq query params
    already in ``neon_db_url`` (``sslmode=require&channel_binding=require``) are
    forwarded to libpq verbatim — no URL surgery, unlike asyncpg.
  * ``prepare_threshold=None`` disables client-side prepared statements, which
    are unsafe under pgbouncer transaction pooling (a prepared statement made on
    one server connection may be absent on the next).
  * ``pool_pre_ping=True`` tolerates Neon dropping idle connections.

Everything is best-effort: if Neon is not configured the engine is simply never
created and ``get_db`` raises a clean 503, mirroring the "degrade gracefully"
convention used by the existing psycopg2 helpers.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singletons, created lazily on first init_engine() (startup).
_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None


def init_engine() -> Optional[AsyncEngine]:
    """Create the async engine + sessionmaker once. Idempotent.

    Returns the engine, or None when no Neon URL is configured (in which case
    the tool-registry endpoints degrade gracefully instead of crashing startup).
    """
    global _engine, _sessionmaker

    if _engine is not None:
        return _engine

    async_url = get_settings().async_database_url
    if not async_url:
        logger.warning(
            "[tools] No neon_db_url configured — Tool Registry async engine not created."
        )
        return None

    _engine = create_async_engine(
        async_url,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        # psycopg v3: never auto-prepare — required for the pgbouncer pooler.
        connect_args={"prepare_threshold": None},
    )
    _sessionmaker = async_sessionmaker(
        _engine, expire_on_commit=False, class_=AsyncSession
    )
    logger.info("[tools] Tool Registry async engine initialised (psycopg v3).")
    return _engine


async def dispose_engine() -> None:
    """Dispose the engine + pool on shutdown. Best-effort; never raises."""
    global _engine, _sessionmaker
    if _engine is not None:
        try:
            await _engine.dispose()
            logger.info("[tools] Tool Registry async engine disposed.")
        except Exception as e:  # pragma: no cover - shutdown best effort
            logger.warning(f"[tools] Engine dispose failed: {e}")
        finally:
            _engine = None
            _sessionmaker = None


def get_engine() -> Optional[AsyncEngine]:
    """Return the engine, creating it on first use if needed."""
    return _engine if _engine is not None else init_engine()


def is_configured() -> bool:
    """True when an async engine is available (Neon configured)."""
    return get_engine() is not None


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yield an AsyncSession, closing it afterwards.

    Raises 503 when the registry DB is not configured so callers get a clear
    signal rather than an opaque AttributeError.
    """
    if _sessionmaker is None:
        init_engine()
    if _sessionmaker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tool Registry storage is not configured (no Neon DB URL).",
        )
    async with _sessionmaker() as session:
        yield session


async def get_optional_db() -> AsyncIterator[Optional[AsyncSession]]:
    """Soft variant of `get_db`: yield ``None`` instead of raising 503 when the
    DB is not configured.

    Used by the read-only Planner so it stays usable (intent + entity resolution
    need no DB) even when Neon is unavailable; DB-dependent steps (tool lookup)
    simply no-op on a ``None`` session.
    """
    if _sessionmaker is None:
        init_engine()
    if _sessionmaker is None:
        yield None
        return
    async with _sessionmaker() as session:
        yield session
