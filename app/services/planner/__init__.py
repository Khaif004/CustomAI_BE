"""Planner package — a READ-ONLY routing layer.

The Planner analyzes a user message and produces a `PlannerResult` (an execution
plan). It NEVER calls the LLM and NEVER generates a response. It reuses existing
services (Tool Registry via `tool_catalog_service`, the in-memory service-tool
registry, the async DB session) and introduces no new architecture.

This module exposes stateless singletons and the FastAPI provider
`get_planner_service`, mirroring how the rest of the backend wires stateless
services. Construction is lazy so importing the package is cheap and free of any
DB/LLM/network side effects.
"""
from __future__ import annotations

from functools import lru_cache

from app.services.planner.entity_resolver import EntityResolver, InMemoryEntityRegistry
from app.services.planner.intent_classifier import IntentClassifier
from app.services.planner.planner_service import PlannerService
from app.services.planner.tool_resolver import ToolCatalogRepository, ToolResolver

__all__ = [
    "PlannerService",
    "IntentClassifier",
    "EntityResolver",
    "ToolResolver",
    "get_planner_service",
]


@lru_cache(maxsize=1)
def get_planner_service() -> PlannerService:
    """FastAPI dependency provider returning the process-wide PlannerService.

    All Planner components are stateless, so a single cached instance is reused
    across requests. The request-scoped AsyncSession is NOT held here — it is
    passed into `analyze(...)` per request (see api/planner_routes.py).
    """
    return PlannerService(
        intent_classifier=IntentClassifier(),
        entity_resolver=EntityResolver(InMemoryEntityRegistry()),
        tool_resolver=ToolResolver(ToolCatalogRepository()),
    )
