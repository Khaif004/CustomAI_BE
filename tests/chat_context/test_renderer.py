"""Unit tests — LLMContext → prepared-context string renderer (temporary adapter)."""
from app.models.llm_context import ContextItem, LLMContext
from app.services.chat_context.renderer import render_llm_context


def test_empty_context_renders_empty_string():
    assert render_llm_context(LLMContext()) == ""


def test_sections_rendered_in_priority_order_with_content():
    ctx = LLMContext(
        live_business_data=[ContextItem(source="LiveOData", title="SalesOrders", content="42 records")],
        application_metadata=[ContextItem(source="Metadata", title="SalesOrder")],
        semantic_knowledge=[ContextItem(source="Pgvector", content="a doc about orders")],
    )
    s = render_llm_context(ctx)
    assert "LIVE BUSINESS DATA" in s
    assert "APPLICATION METADATA" in s
    assert "RETRIEVED KNOWLEDGE" in s
    # exact business data ordered before semantic knowledge
    assert s.index("LIVE BUSINESS DATA") < s.index("RETRIEVED KNOWLEDGE")
    assert "42 records" in s and "a doc about orders" in s


def test_system_instructions_are_not_rendered():
    # SystemInstructions are structured directives for the future Prompt Builder,
    # not prose — the temporary renderer must not emit them.
    ctx = LLMContext(system_instructions=[ContextItem(source="ContextBuilder", title="grounding_policy",
                                                      data={"kind": "grounding_policy"})])
    assert render_llm_context(ctx) == ""
