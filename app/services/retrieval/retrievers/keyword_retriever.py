"""KeywordRetriever — exact keyword matches via PostgreSQL full-text search.

No FTS exists in the codebase, so this computes it on the fly over
`knowledge_documents.content` using `plainto_tsquery` (so arbitrary user text
never raises a parse error), app-scoped via the same `JOIN applications` pattern
the vector search uses. It runs on the async SQLAlchemy session from
`RetrievalRequest` (no executor needed). Degrades gracefully: no session, no
app_id with no global fallback, or any DB error → empty section (errored), never
raises.

NOTE: there is no tsvector column / GIN index (adding one would be DDL on a table
this layer does not own), so this is a sequential scan — acceptable for the
current corpus; a GIN index is a future, separate optimization.
"""
from __future__ import annotations

import hashlib
import logging
from typing import ClassVar, List

from app.models.planner import RetrievalSource
from app.services.retrieval.base import Retriever
from app.services.retrieval.models import (
    TIER_EXACT,
    RetrievalItem,
    RetrievalRequest,
    RetrieverResult,
    Section,
)

logger = logging.getLogger(__name__)

_FTS_APP_SQL = """
SELECT kd.title, kd.content, kd.document_type, kd.metadata,
       ts_rank(to_tsvector('english', kd.content),
               plainto_tsquery('english', :q)) AS rank
FROM knowledge_documents kd
JOIN applications a ON a.id = kd.application_id
WHERE a.application_key = :app
  AND to_tsvector('english', kd.content) @@ plainto_tsquery('english', :q)
ORDER BY rank DESC
LIMIT :k
"""

_FTS_GLOBAL_SQL = """
SELECT kd.title, kd.content, kd.document_type, kd.metadata,
       ts_rank(to_tsvector('english', kd.content),
               plainto_tsquery('english', :q)) AS rank
FROM knowledge_documents kd
WHERE to_tsvector('english', kd.content) @@ plainto_tsquery('english', :q)
ORDER BY rank DESC
LIMIT :k
"""


class KeywordRetriever(Retriever):
    source: ClassVar[RetrievalSource] = RetrievalSource.FULL_TEXT_SEARCH
    section: ClassVar[Section] = Section.KEYWORD_MATCHES

    async def retrieve(self, request: RetrievalRequest) -> RetrieverResult:
        query = (request.message or "").strip()
        if request.session is None or not query:
            return self._empty()
        try:
            from sqlalchemy import text
            if request.app_id:
                sql, params = _FTS_APP_SQL, {"q": query, "app": request.app_id, "k": request.k}
            else:
                sql, params = _FTS_GLOBAL_SQL, {"q": query, "k": request.k}
            rows = (await request.session.execute(text(sql), params)).mappings().all()
        except Exception as e:
            return self._empty(error=str(e))

        items: List[RetrievalItem] = []
        for r in rows:
            content = r.get("content") or ""
            meta = r.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            items.append(RetrievalItem(
                source=self.source, tier=TIER_EXACT,
                score=float(r["rank"]) if r.get("rank") is not None else None,
                ref=hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
                title=r.get("title"),
                content=content,
                data={"category": r.get("document_type"), "metadata": meta},
            ))
        return self._result(items)
