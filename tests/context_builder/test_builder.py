"""Unit tests — ContextBuilder mapping, ranking, provenance, system directives, stats."""
from app.models.llm_context import CONTEXT_BUILDER_SOURCE, ContextBuilderSettings
from app.models.planner import RetrievalSource
from app.services.context_builder.builder import ContextBuilder
from tests.context_builder.helpers import rcontext, ritem


def _builder():
    return ContextBuilder(settings=ContextBuilderSettings())


def test_sources_route_to_correct_sections():
    ctx = rcontext(
        metadata=[ritem(RetrievalSource.METADATA, ref="entity:SalesOrder", title="SalesOrder",
                        data={"kind": "entity", "entity": "SalesOrder", "fields": ["ID"]})],
        tools=[ritem(RetrievalSource.TOOL_REGISTRY, ref="Svc.create", title="create",
                     data={"requiredParameters": ["X"]})],
        live_data=[ritem(RetrievalSource.LIVE_ODATA, ref="live:SalesOrders",
                         data={"entity_set": "SalesOrders", "count": 5})],
        documentation=[ritem(RetrievalSource.DOCUMENTATION, ref="d", content="doc text")],
        conversation_memory=[ritem(RetrievalSource.CONVERSATION_MEMORY, ref="c", content="hi")],
        semantic_documents=[ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s1",
                                  content="semantic text", score=0.7)],
        keyword_matches=[ritem(RetrievalSource.FULL_TEXT_SEARCH, ref="k1", content="keyword text", score=0.4)],
    )
    out = _builder().build(ctx)
    assert len(out.application_metadata) == 1
    assert len(out.tool_metadata) == 1
    assert len(out.live_business_data) == 1
    assert len(out.documentation) == 1
    assert len(out.conversation_context) == 1
    # keyword folds into SemanticKnowledge alongside semantic docs
    assert len(out.semantic_knowledge) == 2
    assert out.current_ui_context == []          # no UI items present → empty, still defined
    assert out.application == "bk"


def test_keyword_exact_ranks_above_semantic_in_semantic_section():
    ctx = rcontext(
        semantic_documents=[ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s1",
                                  content="aaa", score=0.99)],
        keyword_matches=[ritem(RetrievalSource.FULL_TEXT_SEARCH, ref="k1", content="bbb", score=0.1)],
    )
    out = _builder().build(ctx)
    # exact (keyword) outranks semantic even with a far lower score
    assert out.semantic_knowledge[0].source == "FullTextSearch"
    assert out.semantic_knowledge[0].exact is True
    assert out.semantic_knowledge[1].source == "Pgvector"


def test_provenance_preserved():
    ctx = rcontext(
        live_data=[ritem(RetrievalSource.LIVE_ODATA, ref="live:SO",
                         data={"entity_set": "SO", "count": 3, "timestamp": "2026-06-29T10:00:00Z"})],
        semantic_documents=[ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s1",
                                  content="text", score=0.8)],
    )
    out = _builder().build(ctx)
    live = out.live_business_data[0]
    assert live.source == "LiveOData"
    assert live.retriever == "LiveODataRetriever"
    assert live.timestamp == "2026-06-29T10:00:00Z"
    assert live.exact is True
    sem = out.semantic_knowledge[0]
    assert sem.source == "Pgvector" and sem.retriever == "VectorRetriever"
    assert sem.confidence == 0.8                 # falls back to retriever score
    assert sem.exact is False


def test_planner_confidence_used_when_present_in_data():
    ctx = rcontext(
        semantic_documents=[ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s1",
                                  content="t", score=0.2, data={"planner_confidence": 0.97})],
    )
    out = _builder().build(ctx)
    assert out.semantic_knowledge[0].confidence == 0.97


def test_system_directives_are_structured_not_prompts():
    ctx = rcontext(
        metadata=[ritem(RetrievalSource.METADATA, ref="entity:SO", title="SO",
                        data={"kind": "entity", "entity": "SO", "fields": ["ID"]})],
        tools=[ritem(RetrievalSource.TOOL_REGISTRY, ref="Svc.op", title="op", data={})],
        live_data=[ritem(RetrievalSource.LIVE_ODATA, ref="live:SOs", data={"entity_set": "SOs"})],
    )
    out = _builder().build(ctx)
    kinds = {i.data.get("kind") for i in out.system_instructions}
    assert "grounding_policy" in kinds
    assert {"live_data_available", "metadata_available", "tools_available"}.issubset(kinds)
    # data-only: directives carry no prompt prose
    assert all(i.content is None for i in out.system_instructions)
    assert all(i.source == CONTEXT_BUILDER_SOURCE for i in out.system_instructions)


def test_system_directives_can_be_disabled():
    ctx = rcontext(metadata=[ritem(RetrievalSource.METADATA, ref="e", title="E", data={"entity": "E"})])
    out = ContextBuilder(settings=ContextBuilderSettings(include_system_directives=False)).build(ctx)
    assert out.system_instructions == []


def test_statistics_basic():
    ctx = rcontext(
        metadata=[ritem(RetrievalSource.METADATA, ref="e", title="E", data={"entity": "E"})],
        semantic_documents=[ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s", content="x", score=0.5)],
    )
    out = _builder().build(ctx)
    assert out.statistics.token_estimate > 0
    assert "MetadataRetriever" in out.statistics.retrievers_used
    assert "VectorRetriever" in out.statistics.retrievers_used
    # ContextBuilder's own directives are not counted as retrievers
    assert CONTEXT_BUILDER_SOURCE not in out.statistics.retrievers_used
    assert 0.0 < out.statistics.compression_ratio <= 1.0


def test_no_prompt_formatting_only_structured_lists():
    ctx = rcontext(metadata=[ritem(RetrievalSource.METADATA, ref="e", title="E", data={"entity": "E"})])
    out = _builder().build(ctx)
    # Every section is a list of ContextItem — never a concatenated string.
    for section in (out.application_metadata, out.live_business_data, out.semantic_knowledge,
                    out.tool_metadata, out.documentation, out.conversation_context,
                    out.current_ui_context, out.system_instructions):
        assert isinstance(section, list)
