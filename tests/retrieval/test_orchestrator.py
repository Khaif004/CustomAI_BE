"""Unit tests — RetrievalOrchestrator dispatch, isolation, and extensibility."""
from app.models.planner import RetrievalSource
from app.services.retrieval.merger import ResultMerger
from app.services.retrieval.models import RetrievalRequest, Section
from app.services.retrieval.orchestrator import RetrievalOrchestrator
from tests.retrieval.fakes import FakeRetriever, item, make_plan


def _request(plan):
    return RetrievalRequest(message="q", plan=plan, app_id="bk")


async def test_calls_only_required_retrievers():
    meta = FakeRetriever(RetrievalSource.METADATA, Section.METADATA,
                         items=[item(RetrievalSource.METADATA, "e:SalesOrder")])
    vec = FakeRetriever(RetrievalSource.PGVECTOR, Section.SEMANTIC_DOCUMENTS,
                        items=[item(RetrievalSource.PGVECTOR, "d1", tier="semantic")])
    tools = FakeRetriever(RetrievalSource.TOOL_REGISTRY, Section.TOOLS,
                          items=[item(RetrievalSource.TOOL_REGISTRY, "t1")])
    orch = RetrievalOrchestrator([meta, vec, tools], ResultMerger())

    ctx = await orch.retrieve(_request(make_plan([RetrievalSource.METADATA, RetrievalSource.PGVECTOR])))

    assert meta.calls == 1 and vec.calls == 1
    assert tools.calls == 0
    assert len(ctx.metadata) == 1
    assert len(ctx.semantic_documents) == 1
    assert ctx.tools == []
    assert set(ctx.sources_run) == {RetrievalSource.METADATA, RetrievalSource.PGVECTOR}


async def test_unmapped_source_is_skipped():
    meta = FakeRetriever(RetrievalSource.METADATA, Section.METADATA,
                         items=[item(RetrievalSource.METADATA, "e:X")])
    orch = RetrievalOrchestrator([meta], ResultMerger())
    # UI_CONTEXT has no registered retriever → skipped, no error.
    ctx = await orch.retrieve(_request(make_plan([RetrievalSource.METADATA, RetrievalSource.UI_CONTEXT])))
    assert len(ctx.metadata) == 1
    assert ctx.errors == {}


async def test_error_isolation():
    good = FakeRetriever(RetrievalSource.METADATA, Section.METADATA,
                         items=[item(RetrievalSource.METADATA, "e:X")])
    bad = FakeRetriever(RetrievalSource.PGVECTOR, Section.SEMANTIC_DOCUMENTS,
                        raise_exc=RuntimeError("boom"))
    orch = RetrievalOrchestrator([good, bad], ResultMerger())
    ctx = await orch.retrieve(_request(make_plan([RetrievalSource.METADATA, RetrievalSource.PGVECTOR])))
    assert len(ctx.metadata) == 1                       # good retriever unaffected
    assert ctx.semantic_documents == []                 # failed retriever → empty
    assert "Pgvector" in ctx.errors                     # failure recorded, not silent


async def test_open_closed_new_source_without_orchestrator_change():
    # Register a retriever for a source the default set doesn't handle — it runs
    # with ZERO change to the orchestrator class.
    extra = FakeRetriever(RetrievalSource.CODE_SUMMARIES, Section.DOCUMENTATION,
                          items=[item(RetrievalSource.CODE_SUMMARIES, "c1")])
    orch = RetrievalOrchestrator([extra], ResultMerger())
    ctx = await orch.retrieve(_request(make_plan([RetrievalSource.CODE_SUMMARIES])))
    assert len(ctx.documentation) == 1
    assert ctx.documentation[0].source is RetrievalSource.CODE_SUMMARIES


async def test_no_sources_returns_empty_context():
    orch = RetrievalOrchestrator([], ResultMerger())
    ctx = await orch.retrieve(_request(make_plan([])))
    assert ctx.metadata == [] and ctx.tools == [] and ctx.semantic_documents == []
    assert ctx.sources_run == []
