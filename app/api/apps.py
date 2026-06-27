"""
/api/apps — App context registration + OData proxy endpoints.

* POST /api/apps/register   — push entity schema docs into the backend at startup
* POST /api/apps/odata-proxy — proxy an OData request to the originating app
                               using the user's forwarded token, so the agent
                               can fetch real record counts and data without
                               the user needing to know OData or Postman.
"""
from fastapi import APIRouter, HTTPException, status, Depends, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import logging
import re
import urllib.parse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import aiohttp

from app.auth.security import get_current_user
from app.knowledge.knowledge_base import get_knowledge_base

_reg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="btp-reg")

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/apps", tags=["apps"])


class AppDocument(BaseModel):
    title: str = Field(..., description="Short label, e.g. 'SalesOrder entity schema'")
    content: str = Field(..., description="Plain-text content: schema, rules, relationships, etc.")


class AppRegistrationRequest(BaseModel):
    app_id: str = Field(
        ...,
        description="Stable, unique identifier for the app, e.g. 'stutsman'",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    app_name: str = Field(..., description="Human-readable name, e.g. 'Stutsman Sales App'")
    documents: List[AppDocument] = Field(
        ...,
        description="Context documents: entity schemas, relationship descriptions, business rules, etc.",
        min_length=1,
        max_length=2000,
    )
    replace: bool = Field(True, description="Replace previously registered documents for this app_id")


class AppRegistrationResponse(BaseModel):
    app_id: str
    app_name: str
    chunks_stored: int
    docs_received: int
    message: str


@router.post("/register", response_model=AppRegistrationResponse, status_code=status.HTTP_202_ACCEPTED)
async def register_app_context(
    request: AppRegistrationRequest,
):
    """
    Register or update the context for a host application.

    Returns 202 Accepted immediately — embedding and storage happen in a
    background thread so the chat endpoint is never blocked.

    Example payload from Stutsman app:
    ```json
    {
      "app_id": "stutsman",
      "app_name": "Stutsman Sales App",
      "documents": [
        {
          "title": "SalesOrder entity",
          "content": "SalesOrder has fields: id, customerId, createdAt, status (OPEN/CLOSED), totalAmount..."
        },
        {
          "title": "ProcessOrder entity",
          "content": "ProcessOrder has fields: id, salesOrderId (FK), warehouseId, pickedAt, shippedAt..."
        },
        {
          "title": "SalesOrder to ProcessOrder relationship",
          "content": "Each SalesOrder can have one or more ProcessOrders. A SalesOrder transitions to CLOSED only when all its ProcessOrders reach status SHIPPED. ProcessOrder.salesOrderId is a foreign key referencing SalesOrder.id..."
        }
      ]
    }
    ```
    """
    docs = [{"title": d.title, "content": d.content} for d in request.documents]
    app_id = request.app_id
    app_name = request.app_name
    replace = request.replace

    def _do_register():
        try:
            kb = get_knowledge_base()
            result = kb.register_app_context(
                app_id=app_id,
                app_name=app_name,
                documents=docs,
                replace=replace,
            )
            logger.info(
                f"Background registration complete for '{app_id}': "
                f"{result['chunks_stored']} chunks from {result['docs_received']} docs"
            )
        except Exception as exc:
            logger.error(f"Background registration failed for '{app_id}': {exc}", exc_info=True)

    # Fire-and-forget in a dedicated thread pool — does NOT block the event loop
    asyncio.get_event_loop().run_in_executor(_reg_executor, _do_register)

    return AppRegistrationResponse(
        app_id=app_id,
        app_name=app_name,
        chunks_stored=0,
        docs_received=len(docs),
        message=f"Accepted — {len(docs)} documents queued for background indexing.",
    )


# ── Metadata XML Registration ─────────────────────────────────────────────────

_METADATA_DIR = os.path.normpath(os.path.join(__file__, "..", "..", "..", "data", "metadata"))


class MetadataRegistrationRequest(BaseModel):
    app_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    service_url: str = Field(..., description="OData service base URL")
    raw_xml: str = Field(..., description="Raw $metadata XML fetched by the widget")


@router.post("/register-metadata", status_code=status.HTTP_204_NO_CONTENT)
async def register_metadata(request: MetadataRegistrationRequest):
    """
    Store the raw $metadata XML for a service so the agent can inspect it.

    Called by the widget (ContextBridge) once per session — fires in the
    background and returns immediately without blocking chat.
    """
    app_id = request.app_id
    service_url = request.service_url.rstrip("/")
    raw_xml = request.raw_xml

    def _slug(url: str) -> str:
        """Turn a URL into a safe filename slug."""
        slug = re.sub(r"https?://", "", url)
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug)
        return slug[:120]

    def _save():
        try:
            app_dir = os.path.join(_METADATA_DIR, app_id)
            os.makedirs(app_dir, exist_ok=True)
            xml_path = os.path.join(app_dir, f"{_slug(service_url)}.xml")
            with open(xml_path, "w", encoding="utf-8") as fh:
                fh.write(raw_xml)
            logger.info(f"[metadata] Saved $metadata XML for '{app_id}' / '{service_url}' → {xml_path}")
        except Exception as exc:
            logger.error(f"[metadata] Failed to save XML for '{app_id}': {exc}", exc_info=True)

    asyncio.get_event_loop().run_in_executor(_reg_executor, _save)


def load_metadata_xml(app_id: str, service_url: str) -> Optional[str]:
    """
    Load the stored $metadata XML for an app + service.
    Returns None if not found.  Called by the agent at query time.
    """
    def _slug(url: str) -> str:
        slug = re.sub(r"https?://", "", url.rstrip("/"))
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug)
        return slug[:120]

    xml_path = os.path.join(_METADATA_DIR, app_id, f"{_slug(service_url)}.xml")
    try:
        with open(xml_path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.error(f"[metadata] Error reading XML at {xml_path}: {exc}")
        return None


async def deregister_app(app_id: str, current_user=Depends(get_current_user)):
    """Remove all stored context for an app."""
    try:
        kb = get_knowledge_base()
        kb._delete_by_app_id(app_id)
        return {"app_id": app_id, "message": "Context removed."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ── OData Proxy ────────────────────────────────────────────────────────────────

# Allowlist: only relative paths that look like OData entity or function paths.
# Rejects anything that could be used to probe internal infrastructure.
_SAFE_ODATA_PATH = re.compile(
    r"^/odata/v[234]/[a-zA-Z0-9_\-]+/"   # service path
    r"[a-zA-Z0-9_\-]+"                    # entity set or function name
    r"(?:/\$count|\(\d+\))?$",            # optional /$count or (key)
    re.IGNORECASE,
)

_SAFE_QUERY_PARAM = re.compile(
    r"^\$(?:count|top|skip|filter|orderby|select|expand|format)$",
    re.IGNORECASE,
)

_MAX_ROWS = 20   # never return more than 20 rows to the LLM


class ODataProxyRequest(BaseModel):
    """Proxy an OData GET call through the backend so the LLM can access real data."""
    service_url: str = Field(
        ...,
        description="Absolute base URL of the OData service, e.g. http://localhost:4004/odata/v4/fertilizer-blend",
    )
    entity_set: str = Field(
        ...,
        description="OData entity set name, e.g. FertilizerBlend",
        pattern=r"^[a-zA-Z0-9_]+$",
    )
    count_only: bool = Field(
        False,
        description="When true, calls /{entity_set}/$count and returns an integer.",
    )
    filter: Optional[str] = Field(
        None,
        description="$filter expression, e.g. status eq 'OPEN'",
        max_length=500,
    )
    select: Optional[str] = Field(
        None,
        description="Comma-separated $select fields",
        max_length=300,
    )
    top: int = Field(
        5,
        description="Max rows to return (capped at 20)",
        ge=1,
        le=_MAX_ROWS,
    )
    odata_token: Optional[str] = Field(
        None,
        description="Bearer token forwarded from the host Fiori app.",
    )


class ODataProxyResponse(BaseModel):
    entity_set: str
    count: Optional[int] = None
    rows: Optional[List[Dict[str, Any]]] = None
    total_count: Optional[int] = None
    error: Optional[str] = None


@router.post("/odata-proxy", response_model=ODataProxyResponse)
async def odata_proxy(
    request: ODataProxyRequest,
    current_user=Depends(get_current_user),
) -> ODataProxyResponse:
    """
    Proxy a safe OData GET request to the originating Fiori/CAP app.

    The backend calls the OData service on behalf of the user (forwarding
    their XSUAA/JWT token) and returns the result so the LLM can answer
    questions like "how many blends are there?" with a real number.

    Security:
    - Only GET requests are issued — no write operations
    - service_url must start with http/https and resolve to a real OData path
    - entity_set is validated against [a-zA-Z0-9_] only
    - $filter / $select are passed through but length-limited
    - Results are capped at 20 rows
    """
    # Build the target URL
    base = request.service_url.rstrip("/")
    entity = request.entity_set

    # Validate that the base URL looks like an OData path (not an internal host)
    try:
        parsed = urllib.parse.urlparse(base)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("scheme")
        if not parsed.path.startswith("/odata"):
            raise ValueError("path")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid service_url: must be an http(s) OData URL.")

    if request.count_only:
        target = f"{base}/{entity}/$count"
        params: dict = {}
        if request.filter:
            params["$filter"] = request.filter
    else:
        target = f"{base}/{entity}"
        params = {"$top": min(request.top, _MAX_ROWS)}
        if request.filter:
            params["$filter"] = request.filter
        if request.select:
            params["$select"] = request.select
        params["$count"] = "true"

    headers: dict = {
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
    }
    if request.odata_token:
        # Strip any existing "Bearer " prefix before re-adding
        raw_token = request.odata_token.replace("Bearer ", "").replace("bearer ", "")
        headers["Authorization"] = f"Bearer {raw_token}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                target,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 401:
                    return ODataProxyResponse(entity_set=entity, error="Unauthorized — token may have expired.")
                if resp.status == 404:
                    return ODataProxyResponse(entity_set=entity, error=f"Entity set '{entity}' not found at {base}.")
                if resp.status != 200:
                    text = await resp.text()
                    return ODataProxyResponse(entity_set=entity, error=f"OData error {resp.status}: {text[:200]}")

                if request.count_only:
                    text = await resp.text()
                    try:
                        count = int(text.strip())
                    except ValueError:
                        count = None
                    return ODataProxyResponse(entity_set=entity, count=count)

                data = await resp.json()
                rows = data.get("value", [])
                total = data.get("@odata.count")
                return ODataProxyResponse(
                    entity_set=entity,
                    rows=rows[:_MAX_ROWS],
                    total_count=int(total) if total is not None else None,
                )

    except aiohttp.ClientConnectorError as e:
        return ODataProxyResponse(entity_set=entity, error=f"Cannot reach OData service: {e}")
    except Exception as e:
        logger.error(f"OData proxy error: {e}", exc_info=True)
        return ODataProxyResponse(entity_set=entity, error=str(e))


# ── Service Tool Registry ──────────────────────────────────────────────────────
# In-memory map: app_id → List[{ app_name, app_base_url, service_url, entities, registered_at }]
# Persisted to Neon PostgreSQL so restarts don't lose registrations.
# CAP apps re-register on every startup anyway, but this covers the gap
# between backend restart and the first CAP app health-check.
_service_tool_registry: Dict[str, List[Dict[str, Any]]] = {}


def _neon_conn():
    """Return a psycopg2 connection to Neon, or None if not configured."""
    try:
        from app.config import get_settings as _gs
        db_url = _gs().neon_db_url
        if not db_url:
            return None
        import psycopg2
        return psycopg2.connect(db_url)
    except Exception as e:
        logger.warning(f"[registry] Neon connection unavailable: {e}")
        return None


def _persist_service_tool(app_id: str, entry: Dict[str, Any]) -> None:
    """Persist a service tool entry to Neon PostgreSQL (best-effort, never raises)."""
    conn = _neon_conn()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                # Upsert application
                cur.execute(
                    """
                    INSERT INTO applications (application_key, name)
                    VALUES (%s, %s)
                    ON CONFLICT (application_key) DO UPDATE
                        SET name = EXCLUDED.name, updated_at = NOW()
                    RETURNING id
                    """,
                    (app_id, entry.get("app_name", app_id)),
                )
                app_uuid = str(cur.fetchone()[0])

                # Delete existing service entry for this URL (replace approach)
                service_url = entry.get("service_url", "")
                cur.execute(
                    "DELETE FROM services WHERE application_id = %s AND service_url = %s",
                    (app_uuid, service_url),
                )

                # Insert new service row (service_namespace stores app_base_url)
                cur.execute(
                    """
                    INSERT INTO services (application_id, service_url, service_name, service_namespace)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (app_uuid, service_url, entry.get("app_name", app_id),
                     entry.get("app_base_url", "")),
                )
                svc_uuid = str(cur.fetchone()[0])

                import json as _json
                import re as _re
                _ent_fields_map = entry.get("entity_fields") or {}

                for ent in entry.get("entities", []):
                    _fields = _ent_fields_map.get(ent, [])
                    _fk_fields = [f for f in _fields if f.lower().startswith("to_")]
                    _key_fields = [
                        f for f in _fields
                        if not f.lower().startswith("to_")
                        and _re.search(r'(?i)(^id$|ID$|Key$|Number$|Code$)', f)
                        and f not in ("IsActiveEntity", "HasActiveEntity", "HasDraftEntity")
                    ]

                    # ── Insert/upsert entity row ───────────────────────────
                    cur.execute(
                        """
                        INSERT INTO entities (service_id, entity_name, entity_fields, fk_filters)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (service_id, entity_name) DO UPDATE
                            SET entity_fields = EXCLUDED.entity_fields,
                                fk_filters    = EXCLUDED.fk_filters,
                                updated_at    = NOW()
                        RETURNING id
                        """,
                        (svc_uuid, ent,
                         _json.dumps(_fields),
                         _json.dumps(_fk_fields)),
                    )
                    ent_uuid = str(cur.fetchone()[0])

                    # ── Populate normalized entity_fields table ────────────
                    # Delete old rows first (replace approach)
                    cur.execute("DELETE FROM entity_fields WHERE entity_id = %s", (ent_uuid,))
                    for field_name in _fields:
                        _is_key = field_name in _key_fields
                        _is_fk  = field_name.lower().startswith("to_")
                        if _is_fk:
                            _dtype = "Association"
                        elif _re.search(r'(?i)(^id$|ID$)', field_name):
                            _dtype = "Integer"
                        elif field_name in ("createdAt", "modifiedAt"):
                            _dtype = "Timestamp"
                        elif field_name in ("createdBy", "modifiedBy"):
                            _dtype = "String"
                        elif _re.search(r'(?i)(Size$|Count$|Qty$|Quantity$|Amount$|Number$)', field_name):
                            _dtype = "Decimal"
                        else:
                            _dtype = "String"
                        cur.execute(
                            """
                            INSERT INTO entity_fields
                                (entity_id, field_name, data_type, is_key, is_nullable)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (ent_uuid, field_name, _dtype, _is_key,
                             field_name not in _key_fields),
                        )

                    for fk_field in _fk_fields:
                        _fk_m = _re.match(r'to_([A-Za-z][A-Za-z0-9]*)_(\w+)$', fk_field)
                        if not _fk_m:
                            continue
                        parent_ent  = _fk_m.group(1)
                        parent_key  = _fk_m.group(2)
                        _is_int_key = bool(_re.search(r'(?i)(^id$|ID$)', parent_key))
                        cur.execute(
                            """
                            INSERT INTO entity_associations
                                (application_id, source_service_id, source_entity_name,
                                 target_entity_name, fk_field,
                                 relationship_type, cardinality, is_integer_key)
                            VALUES (%s, %s, %s, %s, %s, 'composition', 'many-to-one', %s)
                            ON CONFLICT (source_service_id, source_entity_name, fk_field)
                            DO UPDATE SET
                                target_entity_name = EXCLUDED.target_entity_name,
                                is_integer_key     = EXCLUDED.is_integer_key
                            """,
                            (app_uuid, svc_uuid, ent,
                             parent_ent, fk_field, _is_int_key),
                        )

                # ── Populate entity_aliases (bare → compound) ──────────────
                # e.g. 'Farms' → 'SelectedFarms', 'Fields' → 'SelectedFields'
                _all_ents = list(_ent_fields_map.keys()) + entry.get("entities", [])
                _ent_set  = sorted(set(_all_ents))
                for compound in _ent_set:
                    _c_lower = compound.lower()
                    for bare in _ent_set:
                        if bare == compound:
                            continue
                        _b_lower = bare.lower()
                        # compound ends with bare name (e.g. SelectedFarms ends with farms/farm)
                        _b_sing = _b_lower[:-1] if _b_lower.endswith('s') else _b_lower
                        if _c_lower != _b_lower and (
                            _c_lower.endswith(_b_lower) or _c_lower.endswith(_b_sing)
                        ):
                            # Fetch compound entity id
                            cur.execute(
                                "SELECT id FROM entities WHERE service_id = %s AND entity_name = %s",
                                (svc_uuid, compound),
                            )
                            _row = cur.fetchone()
                            if _row:
                                _cmp_uuid = str(_row[0])
                                cur.execute(
                                    """
                                    INSERT INTO entity_aliases (entity_id, alias)
                                    VALUES (%s, %s)
                                    ON CONFLICT DO NOTHING
                                    """,
                                    (_cmp_uuid, bare),
                                )

        logger.debug(
            f"[registry] Persisted service tool for '{app_id}' / '{service_url}' "
            f"(entities={len(entry.get('entities',[]))}, "
            f"associations written, aliases written)"
        )
    except Exception as e:
        logger.warning(f"[registry] Failed to persist service tool for '{app_id}': {e}")
    finally:
        conn.close()


def load_service_registry_from_db() -> None:
    """
    Populate _service_tool_registry from Neon PostgreSQL on startup.
    Safe to call multiple times — in-memory entries already present are kept.
    """
    conn = _neon_conn()
    if not conn:
        logger.info("[registry] Neon not configured — skipping DB load.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.application_key, a.name,
                       s.id, s.service_url, s.service_namespace, s.created_at
                FROM applications a
                JOIN services s ON s.application_id = a.id
                ORDER BY a.application_key, s.created_at
                """
            )
            rows = cur.fetchall()

            loaded = 0
            for app_key, app_name, svc_id, svc_url, base_url, reg_at in rows:
                # Load entities AND their field lists for this service
                cur.execute(
                    "SELECT entity_name, entity_fields FROM entities WHERE service_id = %s",
                    (svc_id,)
                )
                ent_rows = cur.fetchall()
                entities = [r[0] for r in ent_rows]
                # Reconstruct entity_fields dict: {entity_name: [field, ...]}
                import json as _json
                entity_fields: Dict[str, list] = {}
                for ent_name, ent_flds_raw in ent_rows:
                    try:
                        flds = _json.loads(ent_flds_raw) if isinstance(ent_flds_raw, str) else (ent_flds_raw or [])
                    except Exception:
                        flds = []
                    if flds:
                        entity_fields[ent_name] = flds

                # Load entity_aliases: alias → canonical entity name
                cur.execute(
                    """
                    SELECT ea.alias, e.entity_name
                    FROM entity_aliases ea
                    JOIN entities e ON e.id = ea.entity_id
                    WHERE e.service_id = %s
                    """,
                    (svc_id,)
                )
                aliases: Dict[str, str] = {row[0]: row[1] for row in cur.fetchall()}

                # Load entity_associations: source_entity → {fk_field, target_entity}
                cur.execute(
                    """
                    SELECT source_entity_name, target_entity_name, fk_field, is_integer_key
                    FROM entity_associations
                    WHERE source_service_id = %s
                    """,
                    (svc_id,)
                )
                associations: list = [
                    {
                        "source": r[0], "target": r[1],
                        "fk_field": r[2], "is_integer_key": r[3],
                    }
                    for r in cur.fetchall()
                ]

                entry = {
                    "app_name": app_name or app_key,
                    "service_url": svc_url or "",
                    "entities": entities,
                    "entity_fields": entity_fields,
                    "entity_aliases": aliases,
                    "entity_associations": associations,
                    "app_base_url": base_url or "",
                    "registered_at": reg_at.isoformat() if reg_at else "",
                }

                services = _service_tool_registry.setdefault(app_key, [])
                # Only add if not already present from a live registration
                if not any(s.get("service_url") == svc_url for s in services):
                    services.append(entry)
                    loaded += 1

        logger.info(f"[registry] Loaded {loaded} service tool entries from Neon DB.")
    except Exception as e:
        logger.warning(f"[registry] Could not load registry from DB: {e}")
    finally:
        conn.close()


class ServiceToolRequest(BaseModel):
    app_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    app_name: str = Field(...)
    service_url: str = Field(..., description="Absolute or relative OData service base URL")
    entities: List[str] = Field(default_factory=list)
    app_base_url: Optional[str] = Field(
        None,
        description="Base URL of the CAP server (e.g. http://localhost:4004) used to resolve relative service_url paths",
    )
    entity_fields: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Map of entity name → list of field names. Sent by cap-plugin so the agent can build OData $filter clauses without needing to parse RAG chunks.",
    )


class ServiceToolResponse(BaseModel):
    app_id: str
    service_url: str
    entities_registered: int
    message: str


@router.post("/register-service-tool", response_model=ServiceToolResponse, status_code=status.HTTP_200_OK)
async def register_service_tool(request: ServiceToolRequest):
    """
    Register a CAP OData service as a live-query tool for the LLM agent.

    Called automatically by cap-copilot-sdk on startup. Stores the mapping
    so that when the LLM agent handles a query for this app_id it can call
    the OData service directly for fresh data instead of relying solely on
    the vector store.
    """
    from datetime import datetime, timezone

    entry = {
        "app_name": request.app_name,
        "service_url": request.service_url,
        "entities": request.entities,
        "entity_fields": request.entity_fields or {},
        "app_base_url": request.app_base_url or "",
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    # Upsert: replace existing entry for this service_url, or append a new one.
    services = _service_tool_registry.setdefault(request.app_id, [])
    for i, svc in enumerate(services):
        if svc.get("service_url") == request.service_url:
            services[i] = entry
            break
    else:
        services.append(entry)

    logger.info(
        f"Service tool registered — app_id='{request.app_id}' "
        f"service='{request.service_url}' entities={len(request.entities)}"
    )

    # Persist to Neon DB so the registry survives backend restarts
    asyncio.get_event_loop().run_in_executor(
        _reg_executor, _persist_service_tool, request.app_id, entry
    )

    return ServiceToolResponse(
        app_id=request.app_id,
        service_url=request.service_url,
        entities_registered=len(request.entities),
        message=f"OData service tool registered with {len(request.entities)} entities.",
    )


@router.get("/service-tools", response_model=Dict[str, Any])
async def list_service_tools():
    """Return all currently registered OData service tools."""
    return _service_tool_registry


@router.get("/{app_id}/doc-count")
async def get_doc_count(app_id: str):

    conn = _neon_conn()
    if not conn:
        return {"count": 0}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM knowledge_documents kd
                JOIN applications a ON a.id = kd.application_id
                WHERE a.application_key = %s
                """,
                (app_id,),
            )
            row = cur.fetchone()
            return {"count": int(row[0]) if row else 0}
    except Exception:
        return {"count": 0}
    finally:
        conn.close()



def get_service_tool(app_id: str) -> Optional[Dict[str, Any]]:
    """Utility for agents to look up the first OData service for a given app.
    Use _service_tool_registry[app_id] directly to iterate all services."""
    services = _service_tool_registry.get(app_id)
    return services[0] if services else None
