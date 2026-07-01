"""Pydantic v2 DTOs for the Tool Registry.

Casing follows the SDK contract exactly (see BTP-Copilot-SDK
``cap-plugin/src/tools``):

  * The **envelope** is snake_case — ``{app_id, app_name, sdk_version,
    tool_count, tools[]}`` — matching the SDK's sibling ``/register`` and
    ``/register-service-tool`` payloads and the rest of this backend.
  * Each **tool** and **parameter** object is camelCase — ``toolKey``,
    ``displayName``, ``cdsType``, ``isCollection`` … — because they are the
    SDK's internal ``ToolMetadata`` objects serialized as-is.

So the nested tool models use an automatic camelCase alias generator (with
``populate_by_name=True`` so they also accept snake_case, e.g. when we rebuild
them from DB rows), while the envelope/response models stay plain snake_case.

The full ``ToolType`` vocabulary the SDK reserves (ACTION, FUNCTION + 6 future
types) is accepted so a forward-compatible SDK never gets a 422 — but only
ACTION and FUNCTION are emitted today, and nothing here executes a tool.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class ToolType(str, Enum):
    """Tool types — mirrors the SDK's frozen ToolType enum.

    ACTION and FUNCTION are the primary executed types. UI_ACTION is a
    client-side shortcut that never makes an HTTP call — the executor returns
    a frontendEvent name and the widget dispatches a CustomEvent. The rest are
    reserved by the SDK for future extractors.
    """

    ACTION = "ACTION"
    FUNCTION = "FUNCTION"
    UI_ACTION = "UI_ACTION"
    NAVIGATION = "NAVIGATION"
    REPORT = "REPORT"
    DOCUMENT = "DOCUMENT"
    WORKFLOW = "WORKFLOW"
    UI = "UI"
    API = "API"


class ToolBinding(str, Enum):
    BOUND = "bound"
    UNBOUND = "unbound"


class _CamelModel(BaseModel):
    """Base for the camelCase tool/parameter objects sent by the SDK."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",  # tolerate future SDK fields without breaking
    )


class ToolParameter(_CamelModel):
    name: str
    type: Optional[str] = None
    cds_type: Optional[str] = None          # cdsType
    required: bool = False
    is_collection: bool = False             # isCollection
    length: Optional[int] = None
    description: Optional[str] = None


class ReturnType(_CamelModel):
    type: Optional[str] = None
    is_collection: bool = False             # isCollection
    cds_type: Optional[str] = None          # cdsType
    summary: Optional[str] = None


class RestrictionGrant(_CamelModel):
    grant: List[str] = Field(default_factory=list)
    to: List[str] = Field(default_factory=list)
    where: Optional[str] = None


class Authorization(_CamelModel):
    required_roles: List[str] = Field(default_factory=list)   # requiredRoles
    restrictions: List[RestrictionGrant] = Field(default_factory=list)


class ToolDefinition(_CamelModel):
    tool_key: str = Field(..., min_length=1)                  # toolKey
    tool_type: ToolType                                       # toolType
    binding: Optional[ToolBinding] = None
    name: Optional[str] = None
    display_name: Optional[str] = None                        # displayName
    description: Optional[str] = None
    service_name: Optional[str] = None                        # serviceName
    entity_name: Optional[str] = None                         # entityName
    bound_entity: Optional[str] = None                        # boundEntity
    http_method: Optional[str] = None                         # httpMethod
    http_endpoint: Optional[str] = None                       # httpEndpoint
    parameters: List[ToolParameter] = Field(default_factory=list)
    required_parameters: List[str] = Field(default_factory=list)  # requiredParameters
    return_type: Optional[ReturnType] = None                  # returnType
    authorization: Optional[Authorization] = None
    cds_name: Optional[str] = None                            # cdsName
    frontend_event: Optional[str] = None                      # frontendEvent — UI_ACTION only


# ── Envelope (snake_case — matches the SDK envelope + existing endpoints) ───────

class RegisterToolsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    app_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    app_name: str = Field(...)
    sdk_version: Optional[str] = None
    tool_count: Optional[int] = None
    tools: List[ToolDefinition] = Field(default_factory=list)
    app_base_url: Optional[str] = None


class RegisterToolsResponse(BaseModel):
    app_id: str
    app_name: str
    tools_received: int
    created: int
    updated: int
    unchanged: int
    message: str


class ToolListResponse(BaseModel):
    app_id: str
    tool_count: int
    tools: List[ToolDefinition] = Field(default_factory=list)
