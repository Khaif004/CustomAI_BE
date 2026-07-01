"""Shared factories and fakes for action execution tests."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.models.tool_catalog import (
    Authorization,
    ToolBinding,
    ToolDefinition,
    ToolParameter,
    ToolType,
)
from app.services.action_execution.models import ActionExecutionRequest
from app.services.action_execution.odata_executor import ODataExecutor, ODataRawResponse


def make_tool(
    tool_key: str = "ReleaseProcessOrder",
    tool_type: ToolType = ToolType.ACTION,
    binding: ToolBinding = ToolBinding.UNBOUND,
    http_method: str = "POST",
    http_endpoint: str = "ReleaseProcessOrder",
    service_name: str = "ProcessOrderService",
    entity_name: str = "ProcessOrders",
    parameters: Optional[List[ToolParameter]] = None,
    required_parameters: Optional[List[str]] = None,
    authorization: Optional[Authorization] = None,
) -> ToolDefinition:
    return ToolDefinition(
        tool_key=tool_key,
        tool_type=tool_type,
        binding=binding,
        name=tool_key,
        display_name=tool_key,
        http_method=http_method,
        http_endpoint=http_endpoint,
        service_name=service_name,
        entity_name=entity_name,
        parameters=parameters or [],
        required_parameters=required_parameters or [],
        authorization=authorization,
    )


def make_param(
    name: str,
    type_: str = "String",
    cds_type: str = "cds.String",
    required: bool = False,
    is_collection: bool = False,
) -> ToolParameter:
    return ToolParameter(
        name=name,
        type=type_,
        cds_type=cds_type,
        required=required,
        is_collection=is_collection,
    )


def make_request(
    tool_key: str = "ReleaseProcessOrder",
    app_id: str = "process-orders",
    parameters: Optional[Dict[str, Any]] = None,
    entity_key: Optional[str] = None,
    odata_token: Optional[str] = "Bearer tok",
    user_roles: Optional[List[str]] = None,
) -> ActionExecutionRequest:
    return ActionExecutionRequest(
        app_id=app_id,
        tool_key=tool_key,
        parameters=parameters or {},
        entity_key=entity_key,
        odata_token=odata_token,
        user_id="test-user",
        user_roles=user_roles or [],
    )


class FakeODataExecutor(ODataExecutor):
    """ODataExecutor that returns a pre-canned response without making HTTP calls."""

    def __init__(
        self,
        response: Optional[ODataRawResponse] = None,
        exc: Optional[Exception] = None,
    ) -> None:
        super().__init__(timeout_sec=5)
        self._response = response or ODataRawResponse(
            http_status=200,
            result={"OrderID": "abc", "status": "Released"},
            messages=["Action executed successfully."],
            raw_url="https://fake/odata",
            duration_ms=12.3,
        )
        self._exc = exc
        self.calls: int = 0

    async def execute(self, tool=None, parameters=None, entity_key=None,  # type: ignore[override]
                      service_base_url=None, odata_token=None):
        self.calls += 1
        if self._exc:
            raise self._exc
        return self._response
