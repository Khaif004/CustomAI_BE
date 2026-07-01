"""Unit tests — IntentClassifier (pure, no DB/LLM)."""
import pytest

from app.models.planner import Intent
from app.services.planner.intent_classifier import IntentClassifier
from app.services.planner.text_signals import PlanSignals

_clf = IntentClassifier()


def classify(message, app_id=None, fiori_context=None):
    signals = PlanSignals.build(message, app_id=app_id, fiori_context=fiori_context)
    return _clf.classify(signals)


@pytest.mark.parametrize(
    "message,app_id,expected",
    [
        ("show me all sales orders", "bk", Intent.DATA_QUERY),
        ("how many open orders are there", "bk", Intent.DATA_QUERY),
        ("list the materials by plant", "bk", Intent.DATA_QUERY),
        ("create a new process order", "bk", Intent.TOOL_EXECUTION),
        ("approve purchase order 4711", "bk", Intent.TOOL_EXECUTION),
        ("can you please cancel the order", "bk", Intent.TOOL_EXECUTION),
        ("what fields does Supplier have", "bk", Intent.SCHEMA),
        ("show me the schema of the Material entity", "bk", Intent.SCHEMA),
        ("what is a process order", None, Intent.KNOWLEDGE),
        ("how do I configure pricing", None, Intent.KNOWLEDGE),
        ("navigate to the orders page", "bk", Intent.NAVIGATION),
        ("take me to the supplier screen", "bk", Intent.NAVIGATION),
        ("generate a pdf report", "bk", Intent.DOCUMENTATION),
        ("export this to excel", "bk", Intent.DOCUMENTATION),
        ("where is the createOrder function implemented in the code", None, Intent.CODE_INTELLIGENCE),
        ("hello", None, Intent.GENERAL_CHAT),
        ("thanks, that was helpful", None, Intent.GENERAL_CHAT),
    ],
)
def test_primary_intent(message, app_id, expected):
    result = classify(message, app_id=app_id)
    assert result.intent is expected, f"{message!r} → {result.intent} (scores={result.scores})"


def test_read_verb_vs_mutating_verb_discrimination():
    # 'process' appears as a NOUN here → must stay DATA_QUERY, not TOOL_EXECUTION.
    assert classify("show me all process orders", app_id="bk").intent is Intent.DATA_QUERY
    # leading imperative mutating verb → TOOL_EXECUTION
    assert classify("process the open orders", app_id="bk").intent is Intent.TOOL_EXECUTION


def test_schema_vs_knowledge_boundary():
    assert classify("what fields does Supplier have", app_id="bk").intent is Intent.SCHEMA
    assert classify("what is a supplier", app_id="bk").intent is Intent.KNOWLEDGE


def test_confidence_in_range_and_clear_case_is_confident():
    r = classify("create a new process order", app_id="bk")
    assert 0.0 <= r.confidence <= 0.99
    assert r.confidence >= 0.6  # a clear single-intent message


def test_general_chat_floor():
    r = classify("hello there", app_id=None)
    assert r.intent is Intent.GENERAL_CHAT
    assert 0.2 <= r.confidence <= 0.95


def test_ambiguous_has_lower_confidence_than_dominant():
    dominant = classify("create a new process order", app_id="bk").confidence
    # 'explain the schema' — KNOWLEDGE (explain) vs SCHEMA (schema) compete → lower confidence.
    ambiguous = classify("explain the schema", app_id=None).confidence
    assert ambiguous < dominant
