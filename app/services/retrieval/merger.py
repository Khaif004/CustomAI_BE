"""ResultMerger — assemble retriever outputs into a RetrievalContext.

Responsibilities (and explicit non-responsibilities):
  * Route each RetrieverResult into its named section (sections stay INDEPENDENT —
    nothing is cross-merged).
  * Remove duplicates WITHIN a section (by item.ref, falling back to a content
    hash) — keeping the highest-scoring instance.
  * Preserve source attribution on every item.
  * Rank exact (metadata/keyword/tools/live) above semantic (vector): items sort
    within a section by (tier, score desc), and the context exposes
    `ranked_items()` for an exact-first cross-section view.
  * Record which sources ran and which errored (observability).

It does NOT build prompts, compress, rewrite, or call any model.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, List

from app.models.planner import PlannerResult
from app.services.retrieval.models import (
    TIER_EXACT,
    RetrievalContext,
    RetrievalItem,
    RetrieverResult,
    Section,
)

logger = logging.getLogger(__name__)

# Section → the RetrievalContext attribute it populates.
_SECTION_ATTR = {
    Section.METADATA: "metadata",
    Section.TOOLS: "tools",
    Section.SEMANTIC_DOCUMENTS: "semantic_documents",
    Section.KEYWORD_MATCHES: "keyword_matches",
    Section.LIVE_DATA: "live_data",
    Section.CONVERSATION_MEMORY: "conversation_memory",
    Section.DOCUMENTATION: "documentation",
}


class ResultMerger:
    def merge(
        self,
        results: List[RetrieverResult],
        plan: PlannerResult | None = None,
    ) -> RetrievalContext:
        context = RetrievalContext(application=plan.application if plan else None)

        # Group items by section, preserving each item's source attribution.
        grouped: Dict[Section, List[RetrievalItem]] = {}
        for res in results:
            context.sources_run.append(res.source)
            if res.error:
                context.errors[res.source.value] = res.error
            if res.items:
                grouped.setdefault(res.section, []).extend(res.items)

        for section, items in grouped.items():
            attr = _SECTION_ATTR.get(section)
            if not attr:
                continue
            setattr(context, attr, self._dedup_and_rank(items))

        return context

    @staticmethod
    def _dedup_and_rank(items: List[RetrievalItem]) -> List[RetrievalItem]:
        best: Dict[str, RetrievalItem] = {}
        for item in items:
            key = item.ref or hashlib.sha256(
                (item.content or item.title or "").encode("utf-8")
            ).hexdigest()[:16]
            existing = best.get(key)
            if existing is None or (item.score or 0.0) > (existing.score or 0.0):
                best[key] = item
        # Within a section: exact tier first, then by score desc.
        return sorted(
            best.values(),
            key=lambda i: (0 if i.tier == TIER_EXACT else 1, -(i.score or 0.0)),
        )
