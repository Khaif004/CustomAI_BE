"""Typed exceptions for the Action Execution Engine.

Every error that the engine can raise is represented here so callers can
catch specific conditions without inspecting raw strings.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from app.services.action_execution.models import ValidationFieldError


class ActionExecutionError(Exception):
    """Base class for all action-execution errors."""

    def __init__(self, message: str, code: str = "EXECUTION_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ToolNotFoundError(ActionExecutionError):
    def __init__(self, app_id: str, tool_key: str) -> None:
        super().__init__(
            f"Tool '{tool_key}' not found for app '{app_id}'.",
            code="TOOL_NOT_FOUND",
        )
        self.app_id = app_id
        self.tool_key = tool_key


class AuthorizationError(ActionExecutionError):
    def __init__(self, tool_key: str, required_roles: List[str]) -> None:
        super().__init__(
            f"User is not authorized to execute '{tool_key}'. "
            f"Required role(s): {required_roles}.",
            code="AUTHORIZATION_DENIED",
        )
        self.required_roles = required_roles


class ParameterValidationError(ActionExecutionError):
    def __init__(self, errors: List["ValidationFieldError"]) -> None:
        super().__init__(
            f"Parameter validation failed: {len(errors)} error(s).",
            code="VALIDATION_ERROR",
        )
        self.errors = errors


class EndpointResolutionError(ActionExecutionError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail, code="ENDPOINT_RESOLUTION_ERROR")


class ODataExecutionError(ActionExecutionError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(
            f"OData request failed with HTTP {status_code}: {detail}",
            code="ODATA_ERROR",
        )
        self.status_code = status_code
        self.detail = detail


class ConfigurationError(ActionExecutionError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail, code="CONFIGURATION_ERROR")
