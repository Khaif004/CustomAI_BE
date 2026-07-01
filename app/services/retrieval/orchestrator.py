"""RetrievalOrchestrator — execute (never decide) the plan's retrieval strategy.

It maps `PlannerResult.retrieval_sources` → retrievers via an injected registry,
runs ONLY the required retrievers, concurrently, with per-retriever error
isolation, and hands the raw results to the ResultMerger to produce a
RetrievalContext.

It makes NO planning decisions (it only reads the plan's sources), builds NO
prompts, performs NO compression, and never calls SAP AI Core. New retrievers are
added by registering them (see `default_retrievers`) — this class never changes.
The Chat API must never call retrievers directly; it goes through this orchestrator.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Sequence, Tuple

from app.models.planner import RetrievalSource
from app.services.retrieval.base import Retriever
from app.services.retrieval.merger import ResultMerger
from app.services.retrieval.models import RetrievalContext, RetrievalRequest, RetrieverResult

logger = logging.getLogger(__name__)


class RetrievalOrchestrator:
    def __init__(self, retrievers: Sequence[Retriever], merger: ResultMerger):
        # Build the source → retriever registry from the retrievers' own `source`.
        self._registry: Dict[RetrievalSource, Retriever] = {r.source: r for r in retrievers}
        self._merger = merger

    @property
    def registry(self) -> Dict[RetrievalSource, Retriever]:
        return dict(self._registry)

    async def retrieve(self, request: RetrievalRequest) -> RetrievalContext:
        # Select only the required retrievers (order-preserving, de-duplicated).
        selected: List[Tuple[RetrievalSource, Retriever]] = []
        seen: set = set()
        for src in request.plan.retrieval_sources:
            if src in seen:
                continue
            seen.add(src)
            retriever = self._registry.get(src)
            if retriever is not None:
                selected.append((src, retriever))
            else:
                # Forward-declared source with no backing retriever (e.g. UIContext,
                # CodeSummaries) — skipped, never an error.
                logger.debug("[retrieval] no retriever registered for source %s — skipping", src)

        if not selected:
            return self._merger.merge([], request.plan)

        raw = await asyncio.gather(
            *(self._guarded(src, r, request) for src, r in selected)
        )
        return self._merger.merge(list(raw), request.plan)

    async def _guarded(
        self, src: RetrievalSource, retriever: Retriever, request: RetrievalRequest
    ) -> RetrieverResult:
        """Run one retriever; isolate failures so one never aborts the others."""
        try:
            return await retriever.retrieve(request)
        except Exception as e:  # pragma: no cover - defensive; retrievers self-guard
            logger.warning("[retrieval] retriever %s failed: %s", src, e)
            return RetrieverResult(section=retriever.section, source=src, items=[], error=str(e))
