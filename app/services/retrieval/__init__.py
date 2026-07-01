"""Retrieval package — a READ-ONLY orchestration layer.

The Retrieval Orchestrator EXECUTES the strategy described by a `PlannerResult`
(produced upstream by the Planner) by coordinating individual retrievers, then
merges their outputs into a `RetrievalContext`. It makes NO planning decisions,
builds NO prompts, performs NO compression, and never calls SAP AI Core.

It reuses existing services (Tool Registry, the in-memory metadata registry, the
pgvector knowledge base, the OData plumbing) — it introduces no new architecture
and modifies none of Planner / Tool Registry / Chat APIs / SAP AI Core.

Exposes the stateless singleton provider `get_retrieval_orchestrator` (mirrors
`app.services.planner.get_planner_service`). The request-scoped AsyncSession is
NOT held here — it is passed into `retrieve(...)` per request.
"""
from __future__ import annotations

from functools import lru_cache

from app.services.retrieval.merger import ResultMerger
from app.services.retrieval.orchestrator import RetrievalOrchestrator
from app.services.retrieval.retrievers import default_retrievers

__all__ = ["RetrievalOrchestrator", "get_retrieval_orchestrator"]


@lru_cache(maxsize=1)
def get_retrieval_orchestrator() -> RetrievalOrchestrator:
    """Process-wide stateless orchestrator with the default retriever set wired in.

    To add a future retriever, add it to `default_retrievers()` — the
    orchestrator itself never changes (open/closed).
    """
    return RetrievalOrchestrator(retrievers=default_retrievers(), merger=ResultMerger())
