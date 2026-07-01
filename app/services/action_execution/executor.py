"""ActionExecutionService — orchestrates the full tool execution pipeline.

Pipeline per request
--------------------
  load tool from registry
      ↓
  authorise (required_roles vs user_roles from JWT)
      ↓
  validate parameters (ParameterValidator)
      ↓
  [UI_ACTION short-circuit] → return frontendEvent payload, no HTTP call
      ↓
  load app base_url from applications table (DB)
      ↓
  execute OData request (ODataExecutor)
      ↓
  return ActionExecutionResult
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tool_catalog import ToolDefinition, ToolType
from app.services.action_execution.confirmation_policy import ConfirmationPolicy
from app.services.action_execution.exceptions import (
    ActionExecutionError,
    AuthorizationError,
    ConfigurationError,
    ParameterValidationError,
    ToolNotFoundError,
)
from app.services.action_execution.models import (
    ActionExecutionRequest,
    ActionExecutionResult,
    ExecutionError,
    ToolExecutionStatus,
)
from app.services.action_execution.odata_executor import ODataExecutor
from app.services.action_execution.parameter_validator import ParameterValidator

logger = logging.getLogger("action_execution")


class ActionExecutionService:
    """Stateless orchestrator — safe to use as a module-level singleton."""

    def __init__(
        self,
        validator: Optional[ParameterValidator] = None,
        policy: Optional[ConfirmationPolicy] = None,
        odata_executor: Optional[ODataExecutor] = None,
    ) -> None:
        self._validator = validator or ParameterValidator()
        self._policy = policy or ConfirmationPolicy()
        self._executor = odata_executor or ODataExecutor()

    # ── public API ────────────────────────────────────────────────────────────

    async def execute(
        self,
        request: ActionExecutionRequest,
        session: Optional[AsyncSession] = None,
    ) -> ActionExecutionResult:
        """Execute a registered tool.  Never raises — errors become result fields."""
        t0 = time.monotonic()
        try:
            return await self._run(request, session, t0)

        except ToolNotFoundError as exc:
            return self._error_result(request, ToolExecutionStatus.NOT_FOUND, exc.code, exc.message, t0)

        except AuthorizationError as exc:
            logger.warning(
                "[action_execution] Authorization denied tool='%s' app='%s' user='%s' required=%s",
                request.tool_key, request.app_id, request.user_id, exc.required_roles,
            )
            return self._error_result(request, ToolExecutionStatus.AUTH_ERROR, exc.code, exc.message, t0)

        except ParameterValidationError as exc:
            return ActionExecutionResult(
                status=ToolExecutionStatus.VALIDATION_ERROR,
                tool_key=request.tool_key,
                app_id=request.app_id,
                success=False,
                execution_time_ms=self._ms(t0),
                error=ExecutionError(
                    code=exc.code,
                    message=exc.message,
                    detail="; ".join(e.message for e in exc.errors),
                ),
            )

        except (asyncio.TimeoutError, TimeoutError):
            logger.warning(
                "[action_execution] Timeout tool='%s' app='%s'",
                request.tool_key, request.app_id,
            )
            return self._error_result(
                request, ToolExecutionStatus.TIMEOUT,
                "TIMEOUT", "The OData request timed out.", t0,
            )

        except ActionExecutionError as exc:
            logger.warning(
                "[action_execution] Execution error tool='%s' app='%s' code=%s msg=%s",
                request.tool_key, request.app_id, exc.code, exc.message,
            )
            return self._error_result(request, ToolExecutionStatus.FAILED, exc.code, exc.message, t0)

        except Exception as exc:
            logger.error(
                "[action_execution] Unexpected error tool='%s' app='%s': %s",
                request.tool_key, request.app_id, exc, exc_info=True,
            )
            return self._error_result(
                request, ToolExecutionStatus.FAILED,
                "UNEXPECTED_ERROR", str(exc), t0,
            )

    # ── pipeline ──────────────────────────────────────────────────────────────

    async def _run(
        self,
        request: ActionExecutionRequest,
        session: Optional[AsyncSession],
        t0: float,
    ) -> ActionExecutionResult:
        # 1. Load tool definition from the async Tool Registry
        tool = await self._load_tool(request.app_id, request.tool_key, session)

        # 2. Authorization check (required_roles vs user JWT scopes)
        self._check_authorization(tool, request.user_roles)

        # 3. Parameter validation
        validation = self._validator.validate(tool, request.parameters)
        if not validation.valid:
            raise ParameterValidationError(validation.errors)

        # 4. UI_ACTION short-circuit — dispatch a browser event, no HTTP call
        if tool.tool_type == ToolType.UI_ACTION:
            logger.info(
                "[action_execution] UI_ACTION tool='%s' app='%s' event='%s'",
                request.tool_key, request.app_id, tool.frontend_event,
            )
            return ActionExecutionResult(
                status=ToolExecutionStatus.SUCCESS,
                tool_key=request.tool_key,
                app_id=request.app_id,
                success=True,
                execution_time_ms=self._ms(t0),
                result={
                    "executionType": "UI_ACTION",
                    "frontendEvent": tool.frontend_event,
                    "payload": request.parameters,
                },
            )

        # 5. Confirmation policy (informational — Widget uses this flag)
        needs_confirmation = self._policy.requires_confirmation(tool)

        # 6. Load the app base URL from the applications table (DB)
        if session is None:
            raise ConfigurationError(
                f"Database session required to resolve base URL for app '{request.app_id}'."
            )
        app_base_url = await self._load_app_base_url(request.app_id, session)

        # 7. Execute the OData request
        logger.info(
            "[action_execution] Executing tool='%s' app='%s' user='%s' "
            "binding='%s' method='%s' endpoint='%s'",
            request.tool_key, request.app_id,
            request.user_id or "anon",
            tool.binding, tool.http_method or "POST",
            tool.http_endpoint,
        )

        raw = await self._executor.execute(
            tool=tool,
            parameters=request.parameters,
            entity_key=request.entity_key,
            app_base_url=app_base_url,
            odata_token=request.odata_token,
        )

        duration_ms = self._ms(t0)
        logger.info(
            "[action_execution] Complete tool='%s' app='%s' http_status=%d "
            "duration_ms=%.1f requires_confirmation=%s",
            request.tool_key, request.app_id,
            raw.http_status, duration_ms, needs_confirmation,
        )

        return ActionExecutionResult(
            status=ToolExecutionStatus.SUCCESS,
            tool_key=request.tool_key,
            app_id=request.app_id,
            success=True,
            http_status_code=raw.http_status,
            result=raw.result,
            messages=raw.messages,
            execution_time_ms=duration_ms,
            requires_confirmation=needs_confirmation,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _load_tool(
        self,
        app_id: str,
        tool_key: str,
        session: Optional[AsyncSession],
    ) -> ToolDefinition:
        if session is not None:
            from app.services.tool_catalog_service import get_tool
            tool = await get_tool(session, app_id, tool_key)
            if tool is not None:
                return tool
        raise ToolNotFoundError(app_id, tool_key)

    @staticmethod
    def _check_authorization(
        tool: ToolDefinition,
        user_roles: List[str],
    ) -> None:
        auth = tool.authorization
        if not auth:
            return
        required = auth.required_roles or []
        if not required:
            return
        if not set(user_roles).intersection(required):
            raise AuthorizationError(tool.tool_key, required)

    @staticmethod
    async def _load_app_base_url(app_id: str, session: AsyncSession) -> str:
        """Read applications.base_url from the DB for this app.

        Raises ConfigurationError when the app is unknown or base_url is empty
        so the error surfaces as a typed CONFIGURATION_ERROR result rather than
        an opaque UNEXPECTED_ERROR.
        """
        from sqlalchemy import text as _text
        row = (
            await session.execute(
                _text("SELECT base_url FROM applications WHERE application_key = :app_id"),
                {"app_id": app_id},
            )
        ).first()
        if row is None or not row[0]:
            raise ConfigurationError(
                f"No base_url configured for app '{app_id}'. "
                f"Ensure the CAP application has registered with app_base_url via "
                f"/api/apps/register-tools or /api/apps/register-service-tool."
            )
        return row[0]

    @staticmethod
    def _ms(t0: float) -> float:
        return round((time.monotonic() - t0) * 1000, 2)

    @staticmethod
    def _error_result(
        request: ActionExecutionRequest,
        status: ToolExecutionStatus,
        code: str,
        message: str,
        t0: float,
    ) -> ActionExecutionResult:
        return ActionExecutionResult(
            status=status,
            tool_key=request.tool_key,
            app_id=request.app_id,
            success=False,
            execution_time_ms=round((time.monotonic() - t0) * 1000, 2),
            error=ExecutionError(code=code, message=message),
        )


# ── module-level singleton ────────────────────────────────────────────────────

_singleton: Optional[ActionExecutionService] = None


def get_action_execution_service() -> ActionExecutionService:
    """Return the process-wide ActionExecutionService singleton."""
    global _singleton
    if _singleton is None:
        _singleton = ActionExecutionService()
    return _singleton
