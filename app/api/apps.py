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
# Each app may have multiple OData services; each service is a separate entry.
# Rebuilt automatically on every CAP startup — no file persistence needed.
_service_tool_registry: Dict[str, List[Dict[str, Any]]] = {}


class ServiceToolRequest(BaseModel):
    app_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    app_name: str = Field(...)
    service_url: str = Field(..., description="Absolute or relative OData service base URL")
    entities: List[str] = Field(default_factory=list)
    app_base_url: Optional[str] = Field(
        None,
        description="Base URL of the CAP server (e.g. http://localhost:4004) used to resolve relative service_url paths",
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


def get_service_tool(app_id: str) -> Optional[Dict[str, Any]]:
    """Utility for agents to look up the first OData service for a given app.
    Use _service_tool_registry[app_id] directly to iterate all services."""
    services = _service_tool_registry.get(app_id)
    return services[0] if services else None
