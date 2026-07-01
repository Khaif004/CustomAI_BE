"""DTOs for the Retrieval Orchestrator.

`RetrievalContext` is the merged output consumed later by the Context Builder.
Each of its seven sections is an independent list of `RetrievalItem`s, and every
item preserves its `source` attribution. We deliberately do NOT build prompts or
compress — items are passed through verbatim.

`RetrievalSource` is REUSED from the Planner (`app.models.planner`) — not
duplicated — so the orchestrator maps `PlannerResult.retrieval_sources` straight
to retrievers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from app.models.planner import PlannerResult, RetrievalSource

# Ranking tiers: exact (metadata/keyword/tools/live) outrank semantic (vector).
TIER_EXACT = "exact"
TIER_SEMANTIC = "semantic"


class Section(str, Enum):
    """The independent sections of a RetrievalContext (== field names below)."""

    METADATA = "metadata"
    TOOLS = "tools"
    SEMANTIC_DOCUMENTS = "semantic_documents"
    KEYWORD_MATCHES = "keyword_matches"
    LIVE_DATA = "live_data"
    CONVERSATION_MEMORY = "conversation_memory"
    DOCUMENTATION = "documentation"


class _CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class RetrievalItem(_CamelModel):
    """One attributed retrieval result. Uniform across all sections.

    `content` holds text payloads (doc chunks, formatted summaries); `data` holds
    structured payloads (tool definitions, live OData rows, field/association
    lists). `ref` is a stable per-section dedup key.
    """

    source: RetrievalSource
    tier: str = TIER_EXACT
    score: Optional[float] = None
    ref: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class RetrievalContext(_CamelModel):
    """Merged, source-attributed retrieval output. Sections stay independent."""

    application: Optional[str] = None
    # Exact-tier sections are listed first (ranked above semantic); sections
    # remain independent lists — nothing is cross-merged.
    metadata: List[RetrievalItem] = Field(default_factory=list)
    tools: List[RetrievalItem] = Field(default_factory=list)
    keyword_matches: List[RetrievalItem] = Field(default_factory=list)
    live_data: List[RetrievalItem] = Field(default_factory=list)
    semantic_documents: List[RetrievalItem] = Field(default_factory=list)
    conversation_memory: List[RetrievalItem] = Field(default_factory=list)
    documentation: List[RetrievalItem] = Field(default_factory=list)
    # Observability — which retrievers ran and which failed (no silent failures).
    sources_run: List[RetrievalSource] = Field(default_factory=list)
    errors: Dict[str, str] = Field(default_factory=dict)

    def ranked_items(self) -> List[RetrievalItem]:
        """Flat, cross-section view ordered exact-tier first then by score desc.

        Convenience for a downstream Context Builder; the sections themselves
        remain independent and authoritative.
        """
        everything = (
            self.metadata + self.tools + self.keyword_matches + self.live_data
            + self.semantic_documents + self.conversation_memory + self.documentation
        )
        return sorted(
            everything,
            key=lambda i: (0 if i.tier == TIER_EXACT else 1, -(i.score or 0.0)),
        )


@dataclass
class RetrieverResult:
    """Raw, pre-merge output of a single retriever."""

    section: Section
    source: RetrievalSource
    items: List[RetrievalItem] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class RetrievalRequest:
    """Immutable-ish input handed to every retriever.

    Carries the executed plan (NOT re-derived) plus the request context the
    retrievers need (query text, app, Fiori context, token, DB session). The
    session comes from `Depends(get_optional_db)` and may be None — DB-backed
    retrievers no-op gracefully in that case.
    """

    message: str
    plan: PlannerResult
    app_id: Optional[str] = None
    fiori_context: Optional[Dict[str, Any]] = None
    odata_token: Optional[str] = None
    user_id: Optional[str] = None
    session: Any = None  # sqlalchemy AsyncSession | None (typed Any to avoid import)
    k: int = 5           # top-k for vector / keyword retrievers
    top: int = 20        # live-OData row cap (matches apps._MAX_ROWS)
