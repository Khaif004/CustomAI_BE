"""Tool Registry service — all async DB I/O for the SDK ToolMetadata catalog.

Responsibilities:
  * ``register_tools`` — idempotent registration of a tool set for one app, in a
    single transaction: upsert the application row, then per-tool hash-compare →
    skip unchanged / insert new / update changed, replacing each tool's
    parameters.
  * ``list_tools`` / ``get_tool`` — read back the catalog for an app.

Idempotency: the SDK already hashes its tool list client-side and skips the POST
when nothing changed, but it does NOT send that hash. So we recompute a
deterministic SHA-256 per tool here (``compute_tool_hash``) and compare it to the
stored ``content_hash``; matching tools are left untouched. This makes repeated
registrations cheap and write-free even when the SDK's local cache is cleared.

Reuse, not duplication: the application-row upsert uses the *identical* SQL
contract as the existing sync path (``apps.py:_persist_service_tool`` →
``INSERT INTO applications ... ON CONFLICT (application_key) ... RETURNING id``),
so both drivers write the shared ``applications`` table compatibly.

This module NEVER executes a tool — it only persists and reads metadata.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tool_catalog import (
    Authorization,
    ReturnType,
    ToolDefinition,
    ToolParameter,
)

logger = logging.getLogger(__name__)


# ── Hashing ─────────────────────────────────────────────────────────────────

def compute_tool_hash(tool: ToolDefinition) -> str:
    """Deterministic SHA-256 of a tool's semantic content.

    Uses the camelCase (by_alias) dump with recursively sorted keys so the hash
    is stable regardless of field ordering — the server-side analogue of the
    SDK's canonical hash. Parameter order is preserved (it is meaningful and
    stable in CDS), matching the SDK's own canonicalization.
    """
    canonical = json.dumps(
        tool.model_dump(by_alias=True, exclude_none=False, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_or_none(value: Any) -> Optional[str]:
    """Serialize a dict/list to a JSON string for a JSONB bind param, else None."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


# ── Application row (shared contract with apps.py:_persist_service_tool) ───────

async def _upsert_application(
    session: AsyncSession,
    app_id: str,
    app_name: str,
    app_base_url: Optional[str] = None,
) -> str:
    """Upsert the applications row and return its UUID (as str).

    Mirrors the existing psycopg2 upsert so the two writers stay compatible on
    the shared ``applications`` table. ``base_url`` is updated only when a
    non-null value is supplied (COALESCE keeps the existing value otherwise).
    """
    result = await session.execute(
        text(
            """
            INSERT INTO applications (application_key, name, base_url)
            VALUES (:app_key, :app_name, :base_url)
            ON CONFLICT (application_key) DO UPDATE
                SET name     = EXCLUDED.name,
                    base_url = COALESCE(EXCLUDED.base_url, applications.base_url),
                    updated_at = NOW()
            RETURNING id
            """
        ),
        {"app_key": app_id, "app_name": app_name or app_id, "base_url": app_base_url or None},
    )
    return str(result.scalar_one())


# ── Register ──────────────────────────────────────────────────────────────────

async def register_tools(
    session: AsyncSession,
    app_id: str,
    app_name: str,
    tools: List[ToolDefinition],
    sdk_version: Optional[str] = None,
    app_base_url: Optional[str] = None,
) -> Dict[str, int]:
    """Idempotently register a tool set for one app in a single transaction.

    Returns counts: ``{"created", "updated", "unchanged"}``. Atomic — any error
    rolls the whole batch back.
    """
    created = updated = unchanged = 0

    async with session.begin():
        app_uuid = await _upsert_application(session, app_id, app_name, app_base_url)

        for tool in tools:
            new_hash = compute_tool_hash(tool)

            existing = (
                await session.execute(
                    text(
                        "SELECT id, content_hash FROM tools "
                        "WHERE application_id = :aid AND tool_key = :tk"
                    ),
                    {"aid": app_uuid, "tk": tool.tool_key},
                )
            ).first()

            if existing is not None and existing.content_hash == new_hash:
                unchanged += 1
                continue  # idempotent: identical tool → no write

            tool_id, was_insert = await _upsert_tool(
                session, app_uuid, tool, new_hash, sdk_version
            )

            # Replace-set the parameters (delete + reinsert) — avoids stale rows.
            await session.execute(
                text("DELETE FROM tool_parameters WHERE tool_id = :tid"),
                {"tid": tool_id},
            )
            for ordinal, p in enumerate(tool.parameters):
                await session.execute(
                    text(
                        """
                        INSERT INTO tool_parameters
                            (tool_id, name, type, cds_type, required,
                             is_collection, length, description, ordinal)
                        VALUES
                            (:tid, :name, :type, :cds_type, :required,
                             :is_collection, :length, :description, :ordinal)
                        """
                    ),
                    {
                        "tid": tool_id,
                        "name": p.name,
                        "type": p.type,
                        "cds_type": p.cds_type,
                        "required": p.required,
                        "is_collection": p.is_collection,
                        "length": p.length,
                        "description": p.description,
                        "ordinal": ordinal,
                    },
                )

            if was_insert:
                created += 1
            else:
                updated += 1

    logger.info(
        f"[tools] Registered app_id='{app_id}': "
        f"{created} created, {updated} updated, {unchanged} unchanged "
        f"({len(tools)} received)."
    )
    return {"created": created, "updated": updated, "unchanged": unchanged}


async def _upsert_tool(
    session: AsyncSession,
    app_uuid: str,
    tool: ToolDefinition,
    content_hash: str,
    sdk_version: Optional[str],
) -> tuple[str, bool]:
    """Insert or update one tool row. Returns (tool_id, was_insert).

    The ``(xmax = 0)`` system-column trick distinguishes a fresh INSERT (xmax 0)
    from an ON CONFLICT UPDATE (xmax != 0), so callers can report created vs
    updated without a prior SELECT.
    """
    params = {
        "aid": app_uuid,
        "tk": tool.tool_key,
        "tt": tool.tool_type.value if tool.tool_type else None,
        "bind": tool.binding.value if tool.binding else None,
        "name": tool.name,
        "dn": tool.display_name,
        "desc": tool.description,
        "sn": tool.service_name,
        "en": tool.entity_name,
        "be": tool.bound_entity,
        "hm": tool.http_method,
        "he": tool.http_endpoint,
        "fe": tool.frontend_event,
        "rt": _json_or_none(
            tool.return_type.model_dump(by_alias=True, mode="json")
            if tool.return_type
            else None
        ),
        "auth": _json_or_none(
            tool.authorization.model_dump(by_alias=True, mode="json")
            if tool.authorization
            else None
        ),
        "rp": _json_or_none(tool.required_parameters or []),
        "cn": tool.cds_name,
        "hash": content_hash,
        "sv": sdk_version,
    }

    result = await session.execute(
        text(
            """
            INSERT INTO tools (
                application_id, tool_key, tool_type, binding, name, display_name,
                description, service_name, entity_name, bound_entity, http_method,
                http_endpoint, frontend_event, return_type, authorization_meta,
                required_parameters, cds_name, content_hash, sdk_version
            )
            VALUES (
                :aid, :tk, :tt, :bind, :name, :dn,
                :desc, :sn, :en, :be, :hm,
                :he, :fe, CAST(:rt AS JSONB), CAST(:auth AS JSONB), CAST(:rp AS JSONB),
                :cn, :hash, :sv
            )
            ON CONFLICT (application_id, tool_key) DO UPDATE SET
                tool_type           = EXCLUDED.tool_type,
                binding             = EXCLUDED.binding,
                name                = EXCLUDED.name,
                display_name        = EXCLUDED.display_name,
                description         = EXCLUDED.description,
                service_name        = EXCLUDED.service_name,
                entity_name         = EXCLUDED.entity_name,
                bound_entity        = EXCLUDED.bound_entity,
                http_method         = EXCLUDED.http_method,
                http_endpoint       = EXCLUDED.http_endpoint,
                frontend_event      = EXCLUDED.frontend_event,
                return_type         = EXCLUDED.return_type,
                authorization_meta  = EXCLUDED.authorization_meta,
                required_parameters = EXCLUDED.required_parameters,
                cds_name            = EXCLUDED.cds_name,
                content_hash        = EXCLUDED.content_hash,
                sdk_version         = EXCLUDED.sdk_version,
                updated_at          = NOW()
            RETURNING id, (xmax = 0) AS inserted
            """
        ),
        params,
    )
    row = result.first()
    return str(row.id), bool(row.inserted)


# ── Read ────────────────────────────────────────────────────────────────────

async def list_tools(session: AsyncSession, app_id: str) -> List[ToolDefinition]:
    """Return all tools registered for an app (joined to applications by key)."""
    rows = (
        await session.execute(
            text(
                """
                SELECT t.* FROM tools t
                JOIN applications a ON a.id = t.application_id
                WHERE a.application_key = :app_key
                ORDER BY t.tool_key
                """
            ),
            {"app_key": app_id},
        )
    ).mappings().all()

    if not rows:
        return []

    tool_ids = [r["id"] for r in rows]
    params_by_tool = await _load_parameters(session, tool_ids)
    return [_row_to_tool(r, params_by_tool.get(r["id"], [])) for r in rows]


async def get_tool(
    session: AsyncSession, app_id: str, tool_key: str
) -> Optional[ToolDefinition]:
    """Return a single tool by (app_id, tool_key), or None if absent."""
    row = (
        await session.execute(
            text(
                """
                SELECT t.* FROM tools t
                JOIN applications a ON a.id = t.application_id
                WHERE a.application_key = :app_key AND t.tool_key = :tk
                """
            ),
            {"app_key": app_id, "tk": tool_key},
        )
    ).mappings().first()

    if row is None:
        return None

    params_by_tool = await _load_parameters(session, [row["id"]])
    return _row_to_tool(row, params_by_tool.get(row["id"], []))


async def _load_parameters(
    session: AsyncSession, tool_ids: List[Any]
) -> Dict[Any, List[Dict[str, Any]]]:
    """Fetch parameters for a set of tool ids, grouped by tool_id (order preserved)."""
    if not tool_ids:
        return {}
    rows = (
        await session.execute(
            text(
                """
                SELECT tool_id, name, type, cds_type, required,
                       is_collection, length, description
                FROM tool_parameters
                WHERE tool_id = ANY(:ids)
                ORDER BY tool_id, ordinal
                """
            ),
            {"ids": list(tool_ids)},
        )
    ).mappings().all()

    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["tool_id"], []).append(dict(r))
    return grouped


def _row_to_tool(
    row: Dict[str, Any], param_rows: List[Dict[str, Any]]
) -> ToolDefinition:
    """Rebuild a ToolDefinition from a DB row + its parameter rows.

    JSONB columns come back as Python dicts/lists (psycopg v3 auto-decodes), with
    camelCase keys (we stored them by_alias), so the nested models are rebuilt
    with ``model_validate``. ``populate_by_name=True`` lets the scalar fields be
    set by snake_case field name.
    """
    return ToolDefinition(
        tool_key=row["tool_key"],
        tool_type=row["tool_type"],
        binding=row.get("binding"),
        name=row.get("name"),
        display_name=row.get("display_name"),
        description=row.get("description"),
        service_name=row.get("service_name"),
        entity_name=row.get("entity_name"),
        bound_entity=row.get("bound_entity"),
        http_method=row.get("http_method"),
        http_endpoint=row.get("http_endpoint"),
        parameters=[
            ToolParameter(
                name=p["name"],
                type=p.get("type"),
                cds_type=p.get("cds_type"),
                required=p.get("required", False),
                is_collection=p.get("is_collection", False),
                length=p.get("length"),
                description=p.get("description"),
            )
            for p in param_rows
        ],
        required_parameters=row.get("required_parameters") or [],
        return_type=(
            ReturnType.model_validate(row["return_type"])
            if row.get("return_type")
            else None
        ),
        authorization=(
            Authorization.model_validate(row["authorization_meta"])
            if row.get("authorization_meta")
            else None
        ),
        cds_name=row.get("cds_name"),
        frontend_event=row.get("frontend_event"),
    )
