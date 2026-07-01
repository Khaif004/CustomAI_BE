"""REST API for the Action Execution Engine.

Endpoints
---------
POST /api/apps/{app_id}/actions/{tool_key}/execute
    Execute a registered CAP action or function.

GET  /api/apps/{app_id}/actions/{tool_key}/confirmation
    Check whether a tool requires user confirmation before execution.

Security: the service layer enforces that only URLs stored in the Tool
Registry are ever called — no user-supplied URLs are accepted.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.db.session import get_optional_db
from app.services.action_execution import (
    ActionExecutionRequest,
    ActionExecutionResult,
    get_action_execution_service,
)

router = APIRouter(prefix="/api/apps", tags=["Action Execution"])


# ── request / response models ─────────────────────────────────────────────────

class ExecuteToolRequest(BaseModel):
    """HTTP request body for tool execution."""

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters. Must match the declared parameter schema.",
    )
    entity_key: Optional[str] = Field(
        default=None,
        description="Entity key for bound actions/functions (e.g. a UUID or integer).",
    )
    odata_token: Optional[str] = Field(
        default=None,
        description="Bearer token forwarded from the host Fiori app for OData calls.",
    )


class ConfirmationCheckResponse(BaseModel):
    app_id: str
    tool_key: str
    requires_confirmation: bool


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/{app_id}/actions/{tool_key}/execute",
    response_model=ActionExecutionResult,
    status_code=status.HTTP_200_OK,
    summary="Execute a registered CAP action or function",
)
async def execute_tool(
    app_id: str,
    tool_key: str,
    body: ExecuteToolRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Optional[AsyncSession] = Depends(get_optional_db),
) -> ActionExecutionResult:
    """Execute a registered CAP OData action or function.

    Only tools stored in the Tool Registry can be executed.
    No user-supplied URLs are ever used.
    """
    user_id = (
        current_user.get("user_name")
        or current_user.get("email")
        or current_user.get("sub")
    )

    # XSUAA scopes arrive as a space-separated string or a list
    raw_scope = current_user.get("scope", [])
    user_roles: List[str] = (
        raw_scope.split() if isinstance(raw_scope, str) else list(raw_scope)
    )

    request = ActionExecutionRequest(
        app_id=app_id,
        tool_key=tool_key,
        parameters=body.parameters,
        entity_key=body.entity_key,
        odata_token=body.odata_token,
        user_id=user_id,
        user_roles=user_roles,
    )

    svc = get_action_execution_service()
    return await svc.execute(request, session=db)


@router.get(
    "/{app_id}/actions/{tool_key}/confirmation",
    response_model=ConfirmationCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Check whether a tool requires user confirmation",
)
async def check_confirmation(
    app_id: str,
    tool_key: str,
    _current_user: Dict[str, Any] = Depends(get_current_user),
    db: Optional[AsyncSession] = Depends(get_optional_db),
) -> ConfirmationCheckResponse:
    """Return whether the Widget should display a confirmation dialog before executing."""
    from app.services.action_execution.confirmation_policy import ConfirmationPolicy
    from app.services.tool_catalog_service import get_tool

    requires = False
    if db is not None:
        tool = await get_tool(db, app_id, tool_key)
        if tool is not None:
            requires = ConfirmationPolicy().requires_confirmation(tool)

    return ConfirmationCheckResponse(
        app_id=app_id,
        tool_key=tool_key,
        requires_confirmation=requires,
    )
