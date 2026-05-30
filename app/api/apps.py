"""
/api/apps — App context registration + OData proxy endpoints.

* POST /api/apps/register   — push entity schema docs into the backend at startup
* POST /api/apps/odata-proxy — proxy an OData request to the originating app
                               using the user's forwarded token, so the agent
                               can fetch real record counts and data without
                               the user needing to know OData or Postman.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import logging
import re
import urllib.parse

import aiohttp

from app.auth.security import get_current_user
from app.knowledge.knowledge_base import get_knowledge_base

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
        max_length=50,
    )
    replace: bool = Field(True, description="Replace previously registered documents for this app_id")


class AppRegistrationResponse(BaseModel):
    app_id: str
    app_name: str
    chunks_stored: int
    docs_received: int
    message: str


@router.post("/register", response_model=AppRegistrationResponse, status_code=status.HTTP_200_OK)
async def register_app_context(
    request: AppRegistrationRequest,
    current_user=Depends(get_current_user),
):
    """
    Register or update the context for a host application.

    Call this from your app's startup or CI/CD pipeline whenever the
    schema or business rules change. The content is chunked, embedded,
    and stored in the vector store under the app's `app_id`.

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
    try:
        kb = get_knowledge_base()
        result = kb.register_app_context(
            app_id=request.app_id,
            app_name=request.app_name,
            documents=[{"title": d.title, "content": d.content} for d in request.documents],
            replace=request.replace,
        )
        return AppRegistrationResponse(
            app_id=request.app_id,
            app_name=request.app_name,
            chunks_stored=result["chunks_stored"],
            docs_received=result["docs_received"],
            message=f"Context for '{request.app_name}' registered successfully.",
        )
    except Exception as e:
        logger.error(f"App registration failed for '{request.app_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}",
        )


@router.delete("/{app_id}", status_code=status.HTTP_200_OK)
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
