"""VectorRetriever — semantic documents from the existing pgvector store.

Reuses `get_knowledge_base().vector_store.search(query, k, score_threshold,
metadata_filter={"app_id": app_id})`, which returns structured
`[{content, score, metadata}]`. That search is fully SYNCHRONOUS (psycopg2 +
synchronous embedding HTTP call), so it is offloaded with `asyncio.to_thread` to
avoid blocking the event loop. The embedding call lives entirely inside the KB —
the orchestrator never calls SAP AI Core directly.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Callable, ClassVar, List, Optional

from app.models.planner import RetrievalSource
from app.services.retrieval.base import Retriever
from app.services.retrieval.models import (
    TIER_SEMANTIC,
    RetrievalItem,
    RetrievalRequest,
    RetrieverResult,
    Section,
)

logger = logging.getLogger(__name__)


def _default_kb_provider():
    from app.knowledge.knowledge_base import get_knowledge_base
    return get_knowledge_base()


class VectorRetriever(Retriever):
    source: ClassVar[RetrievalSource] = RetrievalSource.PGVECTOR
    section: ClassVar[Section] = Section.SEMANTIC_DOCUMENTS

    def __init__(self, kb_provider: Optional[Callable] = None):
        # Injected for tests; defaults to the existing KB singleton accessor.
        self._kb_provider = kb_provider or _default_kb_provider

    async def retrieve(self, request: RetrievalRequest) -> RetrieverResult:
        query = (request.message or "").strip()
        if not query:
            return self._empty()
        metadata_filter = {"app_id": request.app_id} if request.app_id else None
        try:
            kb = self._kb_provider()
            rows = await asyncio.to_thread(
                kb.vector_store.search, query, request.k, 0.0, metadata_filter
            )
        except Exception as e:
            return self._empty(error=str(e))

        items: List[RetrievalItem] = []
        for row in rows or []:
            content = row.get("content", "") if isinstance(row, dict) else ""
            meta = row.get("metadata", {}) if isinstance(row, dict) else {}
            if not isinstance(meta, dict):
                meta = {}
            items.append(RetrievalItem(
                source=self.source, tier=TIER_SEMANTIC,
                score=row.get("score") if isinstance(row, dict) else None,
                ref=hashlib.sha256((content or "").encode("utf-8")).hexdigest()[:16],
                title=meta.get("title"),
                content=content,
                data={"metadata": meta},
            ))
        return self._result(items)
