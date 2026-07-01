"""/api/apps — Tool Registry endpoints (SDK ToolMetadata catalog).

A SEPARATE router from ``apps.py`` that shares the same ``/api/apps`` prefix, so
the URLs sit in the existing namespace while the async-SQLAlchemy code stays
physically isolated from the sync psycopg2 code in ``apps.py``. FastAPI merges
the two routers cleanly (no path overlaps).

Endpoints:
  * POST /api/apps/register-tools          — idempotent bulk registration
  * GET  /api/apps/{app_id}/tools          — list an app's tools
  * GET  /api/apps/{app_id}/tools/{tool_key} — fetch a single tool

The SDK (``cap-plugin/src/tools``) posts the snake_case envelope; success is any
HTTP 2xx. Nothing here executes a tool — registration is metadata-only.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.tool_catalog import (
    RegisterToolsRequest,
    RegisterToolsResponse,
    ToolDefinition,
    ToolListResponse,
)
from app.services import tool_catalog_service as svc

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/apps", tags=["tools"])


@router.post(
    "/register-tools",
    response_model=RegisterToolsResponse,
    status_code=status.HTTP_200_OK,
)
async def register_tools(
    request: RegisterToolsRequest,
    db: AsyncSession = Depends(get_db),
) -> RegisterToolsResponse:
    """Register (idempotently) the tools discovered by the SDK for one app.

    Called automatically by cap-copilot-sdk on CAP startup when its tool set has
    changed. Per-tool hash comparison means re-registering an unchanged set is a
    cheap no-op. A genuine DB failure surfaces as 5xx so the SDK's retry/circuit
    breaker can do its job (rather than caching a false success).
    """
    if request.tool_count is not None and request.tool_count != len(request.tools):
        logger.warning(
            f"[tools] tool_count={request.tool_count} != len(tools)={len(request.tools)} "
            f"for app_id='{request.app_id}' — trusting the actual tools array."
        )

    try:
        counts = await svc.register_tools(
            session=db,
            app_id=request.app_id,
            app_name=request.app_name,
            tools=request.tools,
            sdk_version=request.sdk_version,
            app_base_url=request.app_base_url,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[tools] Tool registration failed for app_id='{request.app_id}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tool registration failed.",
        )

    return RegisterToolsResponse(
        app_id=request.app_id,
        app_name=request.app_name,
        tools_received=len(request.tools),
        created=counts["created"],
        updated=counts["updated"],
        unchanged=counts["unchanged"],
        message=(
            f"{counts['created']} created, {counts['updated']} updated, "
            f"{counts['unchanged']} unchanged."
        ),
    )


@router.get("/{app_id}/tools", response_model=ToolListResponse)
async def list_app_tools(
    app_id: str,
    db: AsyncSession = Depends(get_db),
) -> ToolListResponse:
    """List all tools registered for an app (camelCase tool objects out)."""
    tools = await svc.list_tools(db, app_id)
    return ToolListResponse(app_id=app_id, tool_count=len(tools), tools=tools)


@router.get("/{app_id}/tools/{tool_key}", response_model=ToolDefinition)
async def get_app_tool(
    app_id: str,
    tool_key: str,
    db: AsyncSession = Depends(get_db),
) -> ToolDefinition:
    """Fetch a single tool by its stable toolKey (dots allowed, e.g.
    ``CatalogService.Books.addReview``). 404 when the app or tool is unknown."""
    tool = await svc.get_tool(db, app_id, tool_key)
    if tool is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_key}' not found for app '{app_id}'.",
        )
    return tool
