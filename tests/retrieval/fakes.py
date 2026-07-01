"""Test doubles for the Retrieval Orchestrator — no DB, no LLM, no network."""
from __future__ import annotations

from typing import List, Optional

from app.models.planner import Intent, PlannerResult, RetrievalSource
from app.services.retrieval.base import Retriever
from app.services.retrieval.models import RetrievalItem, RetrieverResult, Section


def make_plan(
    sources: List[RetrievalSource],
    *,
    intent: Intent = Intent.DATA_QUERY,
    entity: Optional[str] = None,
    tool: Optional[str] = None,
    requires_live: bool = False,
    application: Optional[str] = "bk",
    confidence: float = 0.9,
) -> PlannerResult:
    return PlannerResult(
        intent=intent,
        confidence=confidence,
        application=application,
        entity=entity,
        tool=tool,
        retrieval_sources=sources,
        requires_live_data=requires_live,
        missing_parameters=[],
    )


class FakeRetriever(Retriever):
    """A retriever whose source/section/items are set per-instance; can raise."""

    def __init__(self, source, section, items=None, raise_exc=None):
        self.source = source
        self.section = section
        self._items = items or []
        self._raise = raise_exc
        self.calls = 0

    async def retrieve(self, request) -> RetrieverResult:
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return RetrieverResult(section=self.section, source=self.source, items=list(self._items))


def item(source: RetrievalSource, ref: str, *, tier: str = "exact", score: float | None = None,
         content: str | None = None, title: str | None = None) -> RetrievalItem:
    return RetrievalItem(source=source, ref=ref, tier=tier, score=score, content=content, title=title)


# ── KB / vector ──────────────────────────────────────────────────────────────

class _FakeVectorStore:
    def __init__(self, rows):
        self._rows = rows
        self.calls = 0

    def search(self, query, k=5, score_threshold=0.0, metadata_filter=None):
        self.calls += 1
        return list(self._rows)


class FakeKB:
    def __init__(self, rows):
        self.vector_store = _FakeVectorStore(rows)


# ── async SQLAlchemy session (for KeywordRetriever) ───────────────────────────

class _FakeMappings:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return _FakeMappings(self._rows)


class FakeSession:
    def __init__(self, rows=None, raise_exc=None):
        self._rows = rows or []
        self._raise = raise_exc
        self.calls = 0

    async def execute(self, *args, **kwargs):
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return _FakeResult(self._rows)


# ── ToolRepository ────────────────────────────────────────────────────────────

class FakeToolRepo:
    def __init__(self, tools=None):
        self._tools = tools or []
        self.calls = 0

    async def list_tools(self, session, app_id):
        self.calls += 1
        return list(self._tools)


# ── Planner (for the route test) ──────────────────────────────────────────────

class FakePlanner:
    def __init__(self, plan: PlannerResult):
        self._plan = plan

    async def analyze(self, message, *, app_id=None, fiori_context=None, session=None):
        return self._plan
