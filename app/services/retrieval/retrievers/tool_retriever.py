"""ToolRetriever — tool metadata from the Tool Registry.

Reuses the existing read API `app.services.tool_catalog_service.list_tools` /
`get_tool` (via the same `ToolCatalogRepository` wrapper the Planner uses). READ
ONLY — never calls register/upsert. No-ops gracefully when there is no DB session
or no app_id. Each tool's full definition (parameters, description, authorization,
required parameters) is preserved verbatim in `data`.
"""
from __future__ import annotations

import logging
from typing import ClassVar, List, Optional

from app.models.planner import RetrievalSource
from app.services.planner.tool_resolver import ToolCatalogRepository, ToolRepository
from app.services.retrieval.base import Retriever
from app.services.retrieval.models import (
    TIER_EXACT,
    RetrievalItem,
    RetrievalRequest,
    RetrieverResult,
    Section,
)

logger = logging.getLogger(__name__)


class ToolRetriever(Retriever):
    source: ClassVar[RetrievalSource] = RetrievalSource.TOOL_REGISTRY
    section: ClassVar[Section] = Section.TOOLS

    def __init__(self, repo: Optional[ToolRepository] = None):
        # Reuse the Planner's repository wrapper over tool_catalog_service by default.
        self._repo = repo or ToolCatalogRepository()

    async def retrieve(self, request: RetrievalRequest) -> RetrieverResult:
        if not request.app_id or request.session is None:
            return self._empty()
        try:
            tools = await self._repo.list_tools(request.session, request.app_id)
        except Exception as e:
            return self._empty(error=str(e))

        # If the plan named a specific tool, surface it first / exclusively.
        planned = request.plan.tool
        items: List[RetrievalItem] = []
        for tool in tools:
            if planned and tool.tool_key != planned:
                continue
            items.append(RetrievalItem(
                source=self.source, tier=TIER_EXACT,
                ref=tool.tool_key,
                title=tool.display_name or tool.name or tool.tool_key,
                content=tool.description,
                data=tool.model_dump(by_alias=True),
            ))
        # If the planned tool wasn't found, fall back to the full list (best-effort).
        if planned and not items:
            for tool in tools:
                items.append(RetrievalItem(
                    source=self.source, tier=TIER_EXACT,
                    ref=tool.tool_key,
                    title=tool.display_name or tool.name or tool.tool_key,
                    content=tool.description,
                    data=tool.model_dump(by_alias=True),
                ))
        return self._result(items)
