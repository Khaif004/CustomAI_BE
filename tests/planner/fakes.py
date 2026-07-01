"""Test doubles for the Planner — no DB, no LLM, no network.

`FakeEntityRegistry` and `FakeToolRepository` implement the same ports the real
adapters do, so the Planner components run fully in-memory under test.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.models.tool_catalog import ToolDefinition, ToolParameter


def make_tool(
    tool_key: str,
    *,
    name: Optional[str] = None,
    tool_type: str = "ACTION",
    entity: Optional[str] = None,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    params: Optional[List[str]] = None,
    required: Optional[List[str]] = None,
) -> ToolDefinition:
    short = tool_key.split(".")[-1]
    required = required or []
    params = params or []
    return ToolDefinition(
        tool_key=tool_key,
        tool_type=tool_type,
        name=name or short,
        display_name=display_name or name or short,
        description=description,
        entity_name=entity,
        bound_entity=entity,
        parameters=[ToolParameter(name=p, required=(p in required)) for p in params],
        required_parameters=list(required),
    )


class FakeEntityRegistry:
    """In-memory EntityRegistry: {app_id: [entity names]} + optional aliases."""

    def __init__(
        self,
        entities_by_app: Optional[Dict[str, List[str]]] = None,
        aliases_by_app: Optional[Dict[str, Dict[str, str]]] = None,
        svc_url: str = "/odata/v4/svc",
    ):
        self._entities = entities_by_app or {}
        self._aliases = aliases_by_app or {}
        self._svc = svc_url

    def get_entities(self, app_id: Optional[str]) -> List[str]:
        return list(self._entities.get(app_id, [])) if app_id else []

    def get_aliases(self, app_id: Optional[str]) -> Dict[str, str]:
        return dict(self._aliases.get(app_id, {})) if app_id else {}

    def service_url_for(self, app_id: Optional[str], entity: str) -> Optional[str]:
        return self._svc


class FakeToolRepository:
    """In-memory ToolRepository: {app_id: [ToolDefinition]}; counts calls so tests
    can assert the DB is NOT consulted for read intents / missing app_id."""

    def __init__(self, tools_by_app: Optional[Dict[str, List[ToolDefinition]]] = None):
        self._tools = tools_by_app or {}
        self.calls = 0

    async def list_tools(self, session, app_id: str) -> List[ToolDefinition]:
        self.calls += 1
        return list(self._tools.get(app_id, []))
