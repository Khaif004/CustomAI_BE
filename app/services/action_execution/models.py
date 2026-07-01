"""Pydantic models for the Action Execution Engine.

All models are model-agnostic (no SAP AI Core, no LLM).
They represent the request/response contract for the execution layer only.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTH_ERROR = "AUTH_ERROR"
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    REQUIRES_CONFIRMATION = "REQUIRES_CONFIRMATION"


class ValidationFieldError(BaseModel):
    """A single parameter validation failure."""

    field: str
    message: str
    expected_type: Optional[str] = None
    received_value: Optional[str] = None


class ValidationResult(BaseModel):
    """Aggregate result of parameter validation."""

    valid: bool
    errors: List[ValidationFieldError] = Field(default_factory=list)


class ExecutionError(BaseModel):
    """Structured error returned inside ActionExecutionResult."""

    code: str
    message: str
    detail: Optional[str] = None
    field: Optional[str] = None


class ActionExecutionRequest(BaseModel):
    """Input contract for the execution service.

    Intentionally transport-agnostic — the REST endpoint, the agent, and
    a future MCP adapter all construct this and call execute().
    """

    app_id: str
    tool_key: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    # For bound actions/functions: the entity key (e.g. "some-uuid-123")
    entity_key: Optional[str] = None
    # Bearer token forwarded from the host Fiori app for OData authentication
    odata_token: Optional[str] = None
    user_id: Optional[str] = None
    # XSUAA scopes / application roles extracted from the JWT
    user_roles: List[str] = Field(default_factory=list)


class ActionExecutionResult(BaseModel):
    """Output contract — returned from the service and exposed via REST."""

    status: ToolExecutionStatus
    tool_key: str
    app_id: str
    success: bool
    http_status_code: Optional[int] = None
    result: Optional[Any] = None
    messages: List[str] = Field(default_factory=list)
    error: Optional[ExecutionError] = None
    execution_time_ms: float = 0.0
    # True when the Widget should show a confirmation dialog before re-executing
    requires_confirmation: bool = False
