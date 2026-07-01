"""Unit tests — chat.py orchestration glue (`_build_prepared_context`).

Covers the three behaviours that guarantee safety + fallback:
  * flag OFF  → returns None (legacy flow, agent does its own retrieval)
  * flag ON   → runs the pipeline and returns the rendered prepared_context
  * any error → returns None (NEVER raises; chat turn / streaming unaffected)

Importing app.api.chat constructs the module-level agent singleton; we monkeypatch
the module-level flag, the agent guard, the pipeline provider, and the DB session
so no real agent/DB/LLM/network is exercised.
"""
import app.api.chat as chatmod
import app.db.session as dbsession
import app.services.chat_context.pipeline as pipemod
from app.models.chat import ChatRequest
from app.models.llm_context import LLMContext
from app.services.chat_context.pipeline import PipelineOutput


async def _none_session():
    yield None


class _FakePipeline:
    def __init__(self, out=None, exc=None):
        self._out = out
        self._exc = exc
        self.ran = False

    async def run(self, cc, *, session=None):
        self.ran = True
        if self._exc is not None:
            raise self._exc
        return self._out


def _output(prepared="RENDERED CONTEXT"):
    return PipelineOutput(
        llm_context=LLMContext(), prepared_context=prepared, intent="DATA_QUERY",
        confidence=0.9, requires_live_data=False, token_estimate=5,
        retrievers_used=["MetadataRetriever"], total_ms=1.0,
    )


def _req():
    return ChatRequest(message="show me orders", app_id="bk")


async def test_flag_off_returns_none(monkeypatch):
    monkeypatch.setattr(chatmod, "_CONTEXT_PIPELINE_ENABLED", False)
    monkeypatch.setattr(chatmod, "chat_agent", object())
    res = await chatmod._build_prepared_context(_req(), user_id="u", session_id="s")
    assert res is None


async def test_flag_on_runs_pipeline_and_returns_prepared(monkeypatch):
    fake = _FakePipeline(out=_output("RENDERED CONTEXT"))
    monkeypatch.setattr(chatmod, "_CONTEXT_PIPELINE_ENABLED", True)
    monkeypatch.setattr(chatmod, "chat_agent", object())
    monkeypatch.setattr(pipemod, "get_chat_pipeline", lambda: fake)
    monkeypatch.setattr(dbsession, "get_optional_db", _none_session)
    res = await chatmod._build_prepared_context(_req(), user_id="u", session_id="s")
    assert res == "RENDERED CONTEXT"
    assert fake.ran is True


async def test_pipeline_error_falls_back_to_none(monkeypatch):
    fake = _FakePipeline(exc=RuntimeError("pipeline boom"))
    monkeypatch.setattr(chatmod, "_CONTEXT_PIPELINE_ENABLED", True)
    monkeypatch.setattr(chatmod, "chat_agent", object())
    monkeypatch.setattr(pipemod, "get_chat_pipeline", lambda: fake)
    monkeypatch.setattr(dbsession, "get_optional_db", _none_session)
    res = await chatmod._build_prepared_context(_req(), user_id="u", session_id="s")
    assert res is None          # failure degrades to legacy flow, never raises


async def test_empty_pipeline_output_returns_empty_string(monkeypatch):
    # Pipeline ran but retrieved nothing → "" (agent skips its own retrieval, injects nothing).
    fake = _FakePipeline(out=_output(""))
    monkeypatch.setattr(chatmod, "_CONTEXT_PIPELINE_ENABLED", True)
    monkeypatch.setattr(chatmod, "chat_agent", object())
    monkeypatch.setattr(pipemod, "get_chat_pipeline", lambda: fake)
    monkeypatch.setattr(dbsession, "get_optional_db", _none_session)
    res = await chatmod._build_prepared_context(_req(), user_id="u", session_id="s")
    assert res == ""


def test_stream_endpoint_preserves_sse_contract_with_pipeline_on(monkeypatch):
    """End-to-end /stream with the flag ON: SSE event schema (chunk→done) is
    byte-identical and the agent receives the pipeline's prepared_context."""
    import json as _json

    from fastapi.testclient import TestClient

    import app.main as mainmod
    from app.auth.security import get_current_user

    class _FakeStreamingAgent:
        model_id = "fake-model"

        def __init__(self):
            self.seen = {}

        async def stream_response(self, **kwargs):
            self.seen.update(kwargs)
            for c in ["Hello", " world"]:
                yield c

        def get_status(self):
            return {"status": "healthy"}

    fake_agent = _FakeStreamingAgent()

    async def _no_doc(_msg):
        return None

    monkeypatch.setattr(chatmod, "chat_agent", fake_agent)
    monkeypatch.setattr(chatmod, "_CONTEXT_PIPELINE_ENABLED", True)
    monkeypatch.setattr(chatmod, "_classify_doc_intent", _no_doc)
    monkeypatch.setattr(chatmod, "_save_chat_to_db_sync", lambda *a, **k: None)
    monkeypatch.setattr(pipemod, "get_chat_pipeline", lambda: _FakePipeline(out=_output("PREP")))
    monkeypatch.setattr(dbsession, "get_optional_db", _none_session)

    mainmod.app.dependency_overrides[get_current_user] = lambda: {"sub": "tester"}
    try:
        client = TestClient(mainmod.app)
        resp = client.post("/api/chat/stream", json={"message": "hi", "app_id": "bk"})
        assert resp.status_code == 200
        events = [
            _json.loads(line[len("data: "):])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        types = [e["type"] for e in events]
        assert "chunk" in types and "done" in types        # SSE schema preserved
        text = "".join(e["content"] for e in events if e["type"] == "chunk")
        assert text == "Hello world"                        # streamed tokens intact
        done = next(e for e in events if e["type"] == "done")
        assert "session_id" in done and done["model"] == "fake-model"   # done contract intact
        # the pipeline's prepared_context reached the agent
        assert fake_agent.seen.get("prepared_context") == "PREP"
    finally:
        mainmod.app.dependency_overrides.pop(get_current_user, None)
