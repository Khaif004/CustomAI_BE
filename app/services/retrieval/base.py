"""The clean `Retriever` interface every retriever implements.

A retriever takes a `RetrievalRequest` and returns a `RetrieverResult` for ITS
single section. It must NEVER raise to the orchestrator for an expected failure
(DB down, OData timeout, etc.) — it should return an errored, empty result
instead; the orchestrator additionally guards every call. Sync/blocking work
(e.g. the pgvector search) must be offloaded inside `retrieve` so the interface
stays uniformly async.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar

from app.models.planner import RetrievalSource
from app.services.retrieval.models import RetrievalRequest, RetrieverResult, Section

logger = logging.getLogger(__name__)


class Retriever(ABC):
    """Base class for all retrievers.

    Subclasses set the `source` (which RetrievalSource they serve) and `section`
    (which RetrievalContext section they fill) class attributes. The orchestrator
    keys its registry off `source`; the merger routes items off `section`.
    """

    source: ClassVar[RetrievalSource]
    section: ClassVar[Section]

    @abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> RetrieverResult:
        """Execute this retriever for the given request and return its result."""
        raise NotImplementedError

    # Convenience helpers for subclasses.
    def _empty(self, error: str | None = None) -> RetrieverResult:
        return RetrieverResult(section=self.section, source=self.source, items=[], error=error)

    def _result(self, items) -> RetrieverResult:
        return RetrieverResult(section=self.section, source=self.source, items=items)
