"""Helpers for Context Builder tests — build RetrievalContext inputs directly.

No DB, no LLM, no network: the Context Builder consumes only a RetrievalContext,
so tests construct one in-memory.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.models.planner import RetrievalSource
from app.services.retrieval.models import RetrievalContext, RetrievalItem


def ritem(
    source: RetrievalSource,
    *,
    ref: Optional[str] = None,
    tier: str = "exact",
    score: Optional[float] = None,
    title: Optional[str] = None,
    content: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> RetrievalItem:
    return RetrievalItem(
        source=source, ref=ref, tier=tier, score=score,
        title=title, content=content, data=data or {},
    )


def rcontext(
    *,
    application: Optional[str] = "bk",
    metadata: Optional[List[RetrievalItem]] = None,
    tools: Optional[List[RetrievalItem]] = None,
    live_data: Optional[List[RetrievalItem]] = None,
    documentation: Optional[List[RetrievalItem]] = None,
    conversation_memory: Optional[List[RetrievalItem]] = None,
    semantic_documents: Optional[List[RetrievalItem]] = None,
    keyword_matches: Optional[List[RetrievalItem]] = None,
) -> RetrievalContext:
    return RetrievalContext(
        application=application,
        metadata=metadata or [],
        tools=tools or [],
        live_data=live_data or [],
        documentation=documentation or [],
        conversation_memory=conversation_memory or [],
        semantic_documents=semantic_documents or [],
        keyword_matches=keyword_matches or [],
    )


class CharTokenizer:
    """Deterministic estimator for budget tests: 1 'token' per character."""

    def estimate(self, text: str) -> int:
        return len(text or "")
