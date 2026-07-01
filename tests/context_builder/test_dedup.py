"""Unit tests — ContextBuilder de-duplication (exact + subsumption)."""
from app.models.planner import RetrievalSource
from app.services.context_builder.builder import ContextBuilder
from tests.context_builder.helpers import rcontext, ritem


def _builder():
    return ContextBuilder()


def test_exact_duplicate_content_kept_once_prefers_exact():
    # Same document text from semantic AND keyword → one survives; exact (keyword) wins.
    ctx = rcontext(
        semantic_documents=[ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s1",
                                  content="identical document text", score=0.9)],
        keyword_matches=[ritem(RetrievalSource.FULL_TEXT_SEARCH, ref="k1",
                               content="identical document text", score=0.3)],
    )
    out = _builder().build(ctx)
    assert len(out.semantic_knowledge) == 1
    assert out.semantic_knowledge[0].source == "FullTextSearch"   # exact preferred
    assert out.statistics.duplicate_count == 1


def test_subsumption_metadata_field_beats_semantic_description():
    # Metadata defines field SalesOrder.BlendID; a semantic doc describing the same
    # field (titled "BlendID") must be dropped — keep the metadata definition only.
    ctx = rcontext(
        metadata=[ritem(RetrievalSource.METADATA, ref="entity:SalesOrder", title="SalesOrder",
                        data={"kind": "entity", "entity": "SalesOrder", "fields": ["BlendID"]})],
        semantic_documents=[
            ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s1", title="BlendID",
                  content="BlendID is the blend identifier field.", score=0.8),
            ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s2", title="Unrelated topic",
                  content="something else entirely", score=0.7),
        ],
    )
    out = _builder().build(ctx)
    sem_titles = [i.title for i in out.semantic_knowledge]
    assert "BlendID" not in sem_titles            # subsumed by metadata
    assert "Unrelated topic" in sem_titles         # unrelated doc kept
    assert out.statistics.duplicate_count >= 1
    assert len(out.application_metadata) == 1


def test_subsumption_entity_title_dropped():
    ctx = rcontext(
        metadata=[ritem(RetrievalSource.METADATA, ref="entity:Material", title="Material",
                        data={"kind": "entity", "entity": "Material", "fields": []})],
        keyword_matches=[ritem(RetrievalSource.FULL_TEXT_SEARCH, ref="k1", title="Material",
                               content="The Material entity", score=0.5)],
    )
    out = _builder().build(ctx)
    assert out.semantic_knowledge == []            # keyword doc titled "Material" subsumed


def test_distinct_items_not_deduped():
    ctx = rcontext(
        semantic_documents=[
            ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s1", content="alpha", score=0.6),
            ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s2", content="beta", score=0.5),
        ],
    )
    out = _builder().build(ctx)
    assert len(out.semantic_knowledge) == 2
    assert out.statistics.duplicate_count == 0
