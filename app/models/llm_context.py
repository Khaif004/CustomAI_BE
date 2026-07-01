"""DTOs for the Context Builder layer.

`LLMContext` is the structured, model-agnostic output the Context Builder
produces from a `RetrievalContext`. It is **purely structured data** — it
contains NO prompt formatting and is unaware of any specific LLM. A future,
model-specific Prompt Builder consumes it.

Every `ContextItem` preserves provenance (source, retriever, confidence,
timestamp) to support explainability. Sections are independent lists.

`ContextBuilderSettings` holds the configurable token budgets (no budget values
are hardcoded in the builder logic — they live here and are injected).

Casing mirrors the rest of the platform (`_CamelModel` → camelCase on the wire,
snake_case field names), consistent with `app/models/planner.py` and the
retrieval models.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# Provenance value used for items the builder derives itself (e.g. directives).
CONTEXT_BUILDER_SOURCE = "ContextBuilder"


class ContextSection(str, Enum):
    """The eight independent sections of an LLMContext (== field names below)."""

    SYSTEM_INSTRUCTIONS = "system_instructions"
    APPLICATION_METADATA = "application_metadata"
    LIVE_BUSINESS_DATA = "live_business_data"
    TOOL_METADATA = "tool_metadata"
    DOCUMENTATION = "documentation"
    CONVERSATION_CONTEXT = "conversation_context"
    CURRENT_UI_CONTEXT = "current_ui_context"
    SEMANTIC_KNOWLEDGE = "semantic_knowledge"


class _CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class ContextItem(_CamelModel):
    """One structured, fully-attributed piece of context.

    Provenance (always present where known): `source` (business source, e.g.
    "LiveOData"), `retriever` (the component that produced it), `confidence`,
    `timestamp`. `exact` marks exact business data vs semantic matches.
    """

    source: str
    retriever: Optional[str] = None
    confidence: Optional[float] = None
    score: Optional[float] = None
    timestamp: Optional[str] = None
    exact: bool = True
    ref: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    token_estimate: int = 0


class ContextStatistics(_CamelModel):
    """Observability + explainability metrics for the build."""

    token_estimate: int = 0
    retrievers_used: List[str] = Field(default_factory=list)
    documents_discarded: int = 0          # dropped to satisfy the token budget
    duplicate_count: int = 0              # dropped by de-duplication
    # kept RETRIEVED tokens / original retrieved tokens (excludes builder-derived
    # system directives, which are not "retrieved"). Bounded in (0, 1]; lower = more
    # trimming. Note: `token_estimate` above DOES include directives, so it will not
    # equal compression_ratio * original — they measure different things by design.
    compression_ratio: float = 1.0


class LLMContext(_CamelModel):
    """Structured, model-agnostic context. NO prompt formatting."""

    application: Optional[str] = None
    system_instructions: List[ContextItem] = Field(default_factory=list)
    application_metadata: List[ContextItem] = Field(default_factory=list)
    live_business_data: List[ContextItem] = Field(default_factory=list)
    tool_metadata: List[ContextItem] = Field(default_factory=list)
    documentation: List[ContextItem] = Field(default_factory=list)
    conversation_context: List[ContextItem] = Field(default_factory=list)
    # Explicit alias: to_camel would yield "currentUiContext" (lowercase i); the
    # spec names this section CurrentUIContext, so we pin the wire name to
    # "currentUIContext" to match downstream consumers. (populate_by_name=True keeps
    # snake_case construction working.)
    current_ui_context: List[ContextItem] = Field(
        default_factory=list, alias="currentUIContext"
    )
    semantic_knowledge: List[ContextItem] = Field(default_factory=list)
    statistics: ContextStatistics = Field(default_factory=ContextStatistics)


class ContextBuilderSettings(_CamelModel):
    """Configurable token budgeting. Defaults here are the ONLY place budget
    numbers live; the builder reads them (never hardcodes). Inject an instance to
    override per deployment/request.

    A per-section cap of `None` means "no section-specific limit" (only the global
    `max_tokens` applies). `max_tokens` is the hard global ceiling.
    """

    max_tokens: int = 8000

    # Per-section caps named by the spec.
    max_metadata_tokens: Optional[int] = 2000
    max_live_data_tokens: Optional[int] = 3000
    max_documentation_tokens: Optional[int] = 1500
    max_conversation_tokens: Optional[int] = 1000

    # Per-section caps for the remaining sections (configurable; default generous).
    max_tool_tokens: Optional[int] = 1500
    max_semantic_tokens: Optional[int] = 1500
    max_ui_tokens: Optional[int] = 500
    # System directives are builder-derived, tiny, and load-bearing (grounding
    # policy); uncapped by default so they're never silently dropped. Set a value
    # to cap them.
    max_system_tokens: Optional[int] = None

    # Behaviour toggles.
    include_system_directives: bool = True

    def cap_for(self, section: ContextSection) -> Optional[int]:
        return {
            ContextSection.APPLICATION_METADATA: self.max_metadata_tokens,
            ContextSection.LIVE_BUSINESS_DATA: self.max_live_data_tokens,
            ContextSection.DOCUMENTATION: self.max_documentation_tokens,
            ContextSection.CONVERSATION_CONTEXT: self.max_conversation_tokens,
            ContextSection.TOOL_METADATA: self.max_tool_tokens,
            ContextSection.SEMANTIC_KNOWLEDGE: self.max_semantic_tokens,
            ContextSection.CURRENT_UI_CONTEXT: self.max_ui_tokens,
            ContextSection.SYSTEM_INSTRUCTIONS: self.max_system_tokens,
        }.get(section)
