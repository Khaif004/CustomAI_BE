"""DocumentationRetriever — placeholder for a future Documentation Engine.

No dedicated documentation corpus/service exists yet (general docs are currently
indexed in pgvector and surfaced via the VectorRetriever). This retriever is
registered so DOCUMENTATION sources are handled uniformly, returning an empty
section today; a future Documentation Engine can be slotted in without changing
the orchestrator.
"""
from __future__ import annotations

from typing import ClassVar

from app.models.planner import RetrievalSource
from app.services.retrieval.base import Retriever
from app.services.retrieval.models import RetrievalRequest, RetrieverResult, Section


class DocumentationRetriever(Retriever):
    source: ClassVar[RetrievalSource] = RetrievalSource.DOCUMENTATION
    section: ClassVar[Section] = Section.DOCUMENTATION

    async def retrieve(self, request: RetrievalRequest) -> RetrieverResult:
        return self._empty()
