"""Unit tests — agents consume `prepared_context` and SKIP their own retrieval.

These verify the core integration guarantee: when the pipeline supplies a
prepared context, each agent uses it and does NOT call its internal retrieval;
when it is None, the legacy retrieval path runs unchanged. Agents are built via
__new__ (bypassing heavy __init__) with the few attributes each method touches
stubbed — no LLM, no DB, no network.
"""
import pytest

from app.agents.chat_agent import ChatAgent
from app.agents.global_agent import GlobalChatAgent
from app.agents.sap_ai_core_agent import SAPAICoreAgent


# ── GlobalChatAgent: pure injection helper (no internal retrieval to skip) ──────

def test_global_apply_prepared_context():
    assert GlobalChatAgent._apply_prepared_context("q", None) == "q"
    assert GlobalChatAgent._apply_prepared_context("q", "") == "q"
    out = GlobalChatAgent._apply_prepared_context("q", "CTX")
    assert "CTX" in out and out.rstrip().endswith("q")


# ── ChatAgent (OpenAI app agent) ────────────────────────────────────────────────

def _chat_agent_stub(calls):
    agent = ChatAgent.__new__(ChatAgent)
    agent.total_requests = 0
    agent.last_request_time = 0.0

    async def _rag(message, app_id):
        calls["rag"] = True
        return "RAGCTX"

    async def _live(message, fiori_context, odata_token):
        calls["live"] = True
        return "LIVECTX"

    agent._fetch_rag_context = _rag
    agent._fetch_live_odata_counts = _live
    agent._format_history = lambda h: []

    captured = {}

    class _Chain:
        async def ainvoke(self, payload):
            captured.update(payload)
            return "answer"

    agent.chain = _Chain()

    class _LLM:
        model_name = "gpt-test"

    agent.llm = _LLM()
    return agent, captured


async def test_chatagent_skips_retrieval_when_prepared_context_supplied():
    calls = {}
    agent, captured = _chat_agent_stub(calls)
    res = await agent.get_response(message="hi", app_id="bk", prepared_context="PREPCTX")
    assert calls.get("rag") is None and calls.get("live") is None   # internal retrieval skipped
    assert "PREPCTX" in captured["system_prompt"]
    assert res["response"] == "answer"


async def test_chatagent_uses_retrieval_when_no_prepared_context():
    calls = {}
    agent, captured = _chat_agent_stub(calls)
    await agent.get_response(message="hi", app_id="bk")   # legacy path
    assert calls.get("rag") is True and calls.get("live") is True
    assert "RAGCTX" in captured["system_prompt"]


# ── SAPAICoreAgent (SAP AI Core app agent) ──────────────────────────────────────

async def test_sapagent_skips_all_internal_retrieval_when_prepared_context():
    agent = SAPAICoreAgent.__new__(SAPAICoreAgent)
    agent.request_count = 0
    calls = {"rag": False, "plan": False, "real": False}

    async def _rag(m, a):
        calls["rag"] = True
        return "RAG"

    async def _plan(*a, **k):
        calls["plan"] = True
        return [{"entity": "X", "filter": "f"}]

    async def _real(*a, **k):
        calls["real"] = True
        return "LIVE"

    agent._fetch_rag_context = _rag
    agent._llm_plan_fetch = _plan
    agent._fetch_real_data = _real
    agent._build_system_message = lambda *a, **k: "SYS"

    class _Auth:
        async def get_token(self):
            raise RuntimeError("STOP-AFTER-CONTEXT")  # stop before the real HTTP call

    agent.auth_client = _Auth()

    # prepared_context provided → all three internal retrieval stages must be skipped
    with pytest.raises(RuntimeError, match="STOP-AFTER-CONTEXT"):
        await agent.get_response(message="hi", app_id="bk", prepared_context="PREPCTX")

    assert calls == {"rag": False, "plan": False, "real": False}
