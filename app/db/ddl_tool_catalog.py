"""DDL for the Tool Registry tables: ``tools`` and ``tool_parameters``.

The backend has no migration framework — existing tables (applications,
services, entities, …) are assumed to be provisioned out-of-band. To keep the
Tool Registry self-contained we create its two tables idempotently at startup
via ``CREATE TABLE IF NOT EXISTS``. We deliberately do NOT touch the
``applications`` table (we only reference it via FK).

Linkage:
  * ``tools.application_id`` → ``applications.id`` (UUID), the same UUID the
    existing psycopg2 upsert returns for an ``application_key`` (the SDK app_id).
  * One row per (application, tool_key); ``tool_key`` is the SDK's stable key.
  * ``content_hash`` is a server-computed SHA-256 over the canonical tool JSON,
    enabling idempotent registration (skip unchanged tools).

``gen_random_uuid()`` is available in core PostgreSQL (>= 13), which Neon runs,
so no extension is required.
"""
from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

TOOLS_DDL = """
CREATE TABLE IF NOT EXISTS tools (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id      UUID NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    tool_key            TEXT NOT NULL,
    tool_type           TEXT NOT NULL,            -- ACTION | FUNCTION (others reserved)
    binding             TEXT,                     -- bound | unbound
    name                TEXT,
    display_name        TEXT,
    description         TEXT,
    service_name        TEXT,
    entity_name         TEXT,
    bound_entity        TEXT,
    http_method         TEXT,
    http_endpoint       TEXT,
    return_type         JSONB,                    -- {type,isCollection,cdsType,summary}
    authorization_meta  JSONB,                    -- {requiredRoles[],restrictions[]} ('authorization' is a reserved SQL word)
    required_parameters JSONB,                    -- string[] (denormalized copy)
    cds_name            TEXT,
    frontend_event      TEXT,                     -- UI_ACTION: CustomEvent name dispatched by widget
    content_hash        TEXT NOT NULL,            -- server-computed per-tool hash
    sdk_version         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_tools_app_toolkey UNIQUE (application_id, tool_key)
);
"""

TOOLS_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS ix_tools_application_id ON tools (application_id);
"""

TOOL_PARAMETERS_DDL = """
CREATE TABLE IF NOT EXISTS tool_parameters (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_id       UUID NOT NULL REFERENCES tools(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    type          TEXT,
    cds_type      TEXT,
    required      BOOLEAN NOT NULL DEFAULT FALSE,
    is_collection BOOLEAN NOT NULL DEFAULT FALSE,
    length        INTEGER,
    description   TEXT,
    ordinal       INTEGER NOT NULL DEFAULT 0,     -- preserve payload order
    CONSTRAINT uq_toolparam_tool_name UNIQUE (tool_id, name)
);
"""

TOOL_PARAMETERS_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS ix_tool_parameters_tool_id ON tool_parameters (tool_id);
"""

_MIGRATE_TOOLS_FRONTEND_EVENT = """
ALTER TABLE IF EXISTS tools
    ADD COLUMN IF NOT EXISTS frontend_event TEXT;
"""

_MIGRATE_APPLICATIONS_BASE_URL = """
ALTER TABLE IF EXISTS applications
    ADD COLUMN IF NOT EXISTS base_url TEXT;
"""

_STATEMENTS = (
    TOOLS_DDL,
    TOOLS_INDEX_DDL,
    TOOL_PARAMETERS_DDL,
    TOOL_PARAMETERS_INDEX_DDL,
    _MIGRATE_TOOLS_FRONTEND_EVENT,
    _MIGRATE_APPLICATIONS_BASE_URL,
)


async def ensure_tool_tables(engine: AsyncEngine) -> bool:
    """Create the tool-registry tables if absent. Best-effort.

    Returns True on success, False on failure (logged, never raised) so startup
    is never blocked by a DDL problem.
    """
    try:
        async with engine.begin() as conn:
            for stmt in _STATEMENTS:
                await conn.execute(text(stmt))
        logger.info("[tools] Ensured tables 'tools' and 'tool_parameters' exist.")
        return True
    except Exception as e:
        logger.warning(f"[tools] Could not ensure tool-registry tables: {e}")
        return False
