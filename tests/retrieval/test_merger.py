"""Unit tests — ResultMerger dedup, attribution, ranking, independence."""
from app.models.planner import RetrievalSource
from app.services.retrieval.merger import ResultMerger
from app.services.retrieval.models import RetrievalItem, RetrieverResult, Section
from tests.retrieval.fakes import make_plan


def _res(section, source, items, error=None):
    return RetrieverResult(section=section, source=source, items=items, error=error)


def test_dedup_within_section_keeps_higher_score():
    a = RetrievalItem(source=RetrievalSource.TOOL_REGISTRY, ref="svc.createOrder", score=0.4)
    b = RetrievalItem(source=RetrievalSource.TOOL_REGISTRY, ref="svc.createOrder", score=0.9)
    ctx = ResultMerger().merge([_res(Section.TOOLS, RetrievalSource.TOOL_REGISTRY, [a, b])])
    assert len(ctx.tools) == 1
    assert ctx.tools[0].score == 0.9


def test_source_attribution_preserved():
    it = RetrievalItem(source=RetrievalSource.METADATA, ref="e:SalesOrder")
    ctx = ResultMerger().merge([_res(Section.METADATA, RetrievalSource.METADATA, [it])])
    assert ctx.metadata[0].source is RetrievalSource.METADATA


def test_sections_stay_independent():
    m = RetrievalItem(source=RetrievalSource.METADATA, ref="e:X")
    d = RetrievalItem(source=RetrievalSource.PGVECTOR, ref="d1", tier="semantic", score=0.5)
    ctx = ResultMerger().merge([
        _res(Section.METADATA, RetrievalSource.METADATA, [m]),
        _res(Section.SEMANTIC_DOCUMENTS, RetrievalSource.PGVECTOR, [d]),
    ])
    assert [i.ref for i in ctx.metadata] == ["e:X"]
    assert [i.ref for i in ctx.semantic_documents] == ["d1"]
    # No cross-contamination.
    assert all(i.source is RetrievalSource.METADATA for i in ctx.metadata)


def test_ranked_view_puts_exact_above_semantic():
    exact = RetrievalItem(source=RetrievalSource.METADATA, ref="e:X", tier="exact", score=0.1)
    semantic = RetrievalItem(source=RetrievalSource.PGVECTOR, ref="d1", tier="semantic", score=0.99)
    ctx = ResultMerger().merge([
        _res(Section.SEMANTIC_DOCUMENTS, RetrievalSource.PGVECTOR, [semantic]),
        _res(Section.METADATA, RetrievalSource.METADATA, [exact]),
    ])
    ranked = ctx.ranked_items()
    # exact (low score) still outranks semantic (high score)
    assert ranked[0].tier == "exact"
    assert ranked[0].ref == "e:X"


def test_errors_recorded_and_sources_run_tracked():
    ctx = ResultMerger().merge(
        [_res(Section.LIVE_DATA, RetrievalSource.LIVE_ODATA, [], error="timeout")],
        plan=make_plan([RetrievalSource.LIVE_ODATA]),
    )
    assert ctx.errors == {"LiveOData": "timeout"}
    assert ctx.sources_run == [RetrievalSource.LIVE_ODATA]
    assert ctx.live_data == []
