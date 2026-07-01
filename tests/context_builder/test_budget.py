"""Unit tests — ContextBuilder token budgeting (per-section caps + global ceiling)."""
from app.models.llm_context import ContextBuilderSettings
from app.models.planner import RetrievalSource
from app.services.context_builder.builder import ContextBuilder
from tests.context_builder.helpers import CharTokenizer, rcontext, ritem


def _builder(settings):
    # CharTokenizer => token_estimate == len(content) for content-only items.
    return ContextBuilder(settings=settings, tokenizer=CharTokenizer())


def _doc(source, ref, n, *, tier="exact", score=0.5):
    # Unique content of exactly n chars (so items are NOT de-duplicated; this test
    # exercises budgeting, not dedup). CharTokenizer => token_estimate == n.
    content = (ref + "|" + "x" * n)[:n]
    return ritem(source, ref=ref, tier=tier, content=content, score=score)


def test_per_section_cap_trims_lowest_ranked_first():
    # 3 live items of 100 chars each; cap 250 → keep 2 (200), drop 1.
    s = ContextBuilderSettings(max_tokens=100000, max_live_data_tokens=250,
                               include_system_directives=False)
    ctx = rcontext(live_data=[
        _doc(RetrievalSource.LIVE_ODATA, "l1", 100, score=0.9),
        _doc(RetrievalSource.LIVE_ODATA, "l2", 100, score=0.5),
        _doc(RetrievalSource.LIVE_ODATA, "l3", 100, score=0.1),
    ])
    out = _builder(s).build(ctx)
    assert len(out.live_business_data) == 2
    assert out.statistics.documents_discarded == 1
    # highest-score items survive
    refs = {i.ref for i in out.live_business_data}
    assert refs == {"l1", "l2"}


def test_global_budget_trims_lowest_priority_section_first():
    # Global cap forces trimming; semantic (lowest priority) must go before metadata.
    s = ContextBuilderSettings(max_tokens=150, max_metadata_tokens=None,
                               max_semantic_tokens=None, include_system_directives=False)
    ctx = rcontext(
        metadata=[_doc(RetrievalSource.METADATA, "m1", 100, score=0.9)],
        semantic_documents=[_doc(RetrievalSource.PGVECTOR, "s1", 100, tier="semantic", score=0.9)],
    )
    out = _builder(s).build(ctx)
    assert len(out.application_metadata) == 1       # metadata (higher priority) kept
    assert out.semantic_knowledge == []             # semantic trimmed first
    assert out.statistics.documents_discarded == 1


def test_live_data_kept_over_metadata_under_pressure():
    s = ContextBuilderSettings(max_tokens=120, max_metadata_tokens=None,
                               max_live_data_tokens=None, include_system_directives=False)
    ctx = rcontext(
        live_data=[_doc(RetrievalSource.LIVE_ODATA, "l1", 100, score=0.5)],
        metadata=[_doc(RetrievalSource.METADATA, "m1", 100, score=0.9)],
    )
    out = _builder(s).build(ctx)
    # Only ~120 budget; live (priority 1) survives, metadata (priority 3) trimmed.
    assert len(out.live_business_data) == 1
    assert out.application_metadata == []


def test_compression_ratio_reflects_trimming():
    s = ContextBuilderSettings(max_tokens=100, include_system_directives=False)
    ctx = rcontext(
        semantic_documents=[
            _doc(RetrievalSource.PGVECTOR, "s1", 100, tier="semantic", score=0.9),
            _doc(RetrievalSource.PGVECTOR, "s2", 100, tier="semantic", score=0.8),
        ],
    )
    out = _builder(s).build(ctx)
    # 200 in, ~100 kept → ratio ~0.5
    assert out.statistics.compression_ratio < 1.0
    assert out.statistics.documents_discarded == 1


def test_no_trimming_when_within_budget():
    s = ContextBuilderSettings(max_tokens=100000, include_system_directives=False)
    ctx = rcontext(metadata=[_doc(RetrievalSource.METADATA, "m1", 50, score=0.9)])
    out = _builder(s).build(ctx)
    assert out.statistics.documents_discarded == 0
    assert out.statistics.compression_ratio == 1.0
