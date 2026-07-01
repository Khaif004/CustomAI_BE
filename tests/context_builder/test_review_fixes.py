"""Regression tests for the adversarial-review fixes (M1, M2, S1, S2, S3)."""
from app.models.planner import RetrievalSource
from app.services.context_builder.builder import _SECTION_PRIORITY, _TRIM_ORDER, ContextBuilder
from tests.context_builder.helpers import rcontext, ritem


# ── M1: global trim order is descending priority; live data trimmed LAST ────────

def test_trim_order_is_descending_priority_with_live_last():
    priorities = [_SECTION_PRIORITY[s] for s in _TRIM_ORDER]
    assert priorities == sorted(priorities, reverse=True)        # monotonically descending
    assert _TRIM_ORDER[-1].value == "live_business_data"          # most authoritative kept longest


# ── M2: structured ref is per-section → no cross-section collision ──────────────

def test_cross_section_shared_ref_not_collapsed():
    ctx = rcontext(
        metadata=[ritem(RetrievalSource.METADATA, ref="SalesOrder", title="SalesOrder",
                        data={"kind": "entity", "entity": "SalesOrder"})],
        tools=[ritem(RetrievalSource.TOOL_REGISTRY, ref="SalesOrder", title="SalesOrder action",
                     data={})],
    )
    out = ContextBuilder().build(ctx)
    assert len(out.application_metadata) == 1
    assert len(out.tool_metadata) == 1                            # NOT collapsed across sections
    assert out.statistics.duplicate_count == 0


# ── S2: subsumption fires for realistic (sentence) doc titles ───────────────────

def test_subsumption_matches_sentence_title_but_not_unrelated():
    ctx = rcontext(
        metadata=[ritem(RetrievalSource.METADATA, ref="entity:SalesOrder", title="SalesOrder",
                        data={"kind": "entity", "entity": "SalesOrder", "fields": ["Status"]})],
        semantic_documents=[
            ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s1",
                  title="The Status field indicates order state",
                  content="status doc body", score=0.8),
            ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s2",
                  title="General onboarding guide",
                  content="onboarding doc body", score=0.7),
        ],
    )
    out = ContextBuilder().build(ctx)
    titles = [i.title for i in out.semantic_knowledge]
    assert "The Status field indicates order state" not in titles   # subsumed via 'Status'
    assert "General onboarding guide" in titles                      # unrelated kept
    assert out.statistics.duplicate_count >= 1


def test_subsumption_does_not_overmatch_short_tokens():
    # Field 'ID' (len 2) must NOT subsume a doc merely mentioning "id".
    ctx = rcontext(
        metadata=[ritem(RetrievalSource.METADATA, ref="entity:X", title="X",
                        data={"kind": "entity", "entity": "X", "fields": ["ID"]})],
        semantic_documents=[ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s1",
                                  title="How to validate an id quickly", content="...", score=0.6)],
    )
    out = ContextBuilder().build(ctx)
    assert len(out.semantic_knowledge) == 1                          # not over-pruned


# ── S1: CurrentUIContext is a forward-declared, currently-empty section ──────────

def test_current_ui_context_forward_declared_empty():
    ctx = rcontext(metadata=[ritem(RetrievalSource.METADATA, ref="entity:E", title="E",
                                   data={"entity": "E"})])
    out = ContextBuilder().build(ctx)
    assert out.current_ui_context == []     # no UI retriever exists yet → always empty


# ── S3: wire name matches the spec's CurrentUIContext (currentUIContext) ─────────

def test_current_ui_context_wire_name():
    out = ContextBuilder().build(rcontext())
    dumped = out.model_dump(by_alias=True)
    assert "currentUIContext" in dumped
    assert "currentUiContext" not in dumped
