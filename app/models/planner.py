"""Pydantic v2 DTOs for the Planner layer.

The Planner is a READ-ONLY routing layer: it analyzes a user message and returns
an execution plan (`PlannerResult`). It NEVER generates a response and NEVER
calls the LLM. These models carry only the plan.

Output casing matches the spec example exactly (camelCase): ``intent``,
``confidence``, ``application``, ``entity``, ``tool``, ``retrievalSources``,
``requiresLiveData``, ``missingParameters``. We achieve this with an automatic
camelCase alias generator while keeping Pythonic snake_case field names
(consistent with ``app/models/tool_catalog.py``). FastAPI serializes responses
by alias by default, so the wire shape is camelCase.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class Intent(str, Enum):
    """The primary intent the Planner assigns to a user message."""

    DATA_QUERY = "DATA_QUERY"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    KNOWLEDGE = "KNOWLEDGE"
    SCHEMA = "SCHEMA"
    NAVIGATION = "NAVIGATION"
    DOCUMENTATION = "DOCUMENTATION"
    CODE_INTELLIGENCE = "CODE_INTELLIGENCE"
    GENERAL_CHAT = "GENERAL_CHAT"


class RetrievalSource(str, Enum):
    """Retrieval sources the Planner may declare as required for a request.

    Values match the platform's source vocabulary. The Planner only DECLARES the
    intended sources; it does not itself retrieve. Some sources (e.g.
    CODE_SUMMARIES) may be forward-declared before a backing service exists.
    """

    METADATA = "Metadata"                       # Metadata Repository
    TOOL_REGISTRY = "ToolRegistry"              # Tool Registry
    PGVECTOR = "Pgvector"                       # pgvector semantic store
    FULL_TEXT_SEARCH = "FullTextSearch"         # PostgreSQL full-text search
    LIVE_ODATA = "LiveOData"                    # Live OData calls
    CONVERSATION_MEMORY = "ConversationMemory"  # Prior conversation turns
    UI_CONTEXT = "UIContext"                    # Current Fiori UI context
    DOCUMENTATION = "Documentation"             # Documentation corpus
    CODE_SUMMARIES = "CodeSummaries"            # Code summaries / repo index


class _CamelModel(BaseModel):
    # NOTE: no use_enum_values — keep `intent`/`retrievalSources` as real enum
    # members for downstream consumers; FastAPI/json still serializes them to
    # their string values on the wire.
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class PlannerAnalyzeRequest(_CamelModel):
    """Input to the Planner. Intentionally slim and INDEPENDENT of ChatRequest so
    the Planner never couples to the chat contract. Accepts both snake_case and
    camelCase keys (``app_id``/``appId``, ``fiori_context``/``fioriContext``)."""

    message: str = Field(..., min_length=1, max_length=500000)
    app_id: Optional[str] = Field(None, pattern=r"^[a-zA-Z0-9_-]*$")
    fiori_context: Optional[Dict[str, Any]] = None


class PlannerResult(_CamelModel):
    """The execution plan produced by the Planner. No response text, ever."""

    intent: Intent
    confidence: float = Field(..., ge=0.0, le=1.0)
    application: Optional[str] = None
    entity: Optional[str] = None
    tool: Optional[str] = None
    retrieval_sources: List[RetrievalSource] = Field(default_factory=list)
    requires_live_data: bool = False
    missing_parameters: List[str] = Field(default_factory=list)
