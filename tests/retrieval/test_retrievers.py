"""Unit tests — individual retrievers (fakes; no DB/LLM/network)."""
import app.api.apps as apps_mod
from app.models.planner import RetrievalSource
from app.services.retrieval.models import RetrievalRequest
from app.services.retrieval.retrievers.documentation_retriever import DocumentationRetriever
from app.services.retrieval.retrievers.keyword_retriever import KeywordRetriever
from app.services.retrieval.retrievers.live_odata_retriever import LiveODataRetriever
from app.services.retrieval.retrievers.memory_retriever import MemoryRetriever
from app.services.retrieval.retrievers.metadata_retriever import MetadataRetriever
from app.services.retrieval.retrievers.tool_retriever import ToolRetriever
from app.services.retrieval.retrievers.vector_retriever import VectorRetriever
from tests.planner.fakes import make_tool
from tests.retrieval.fakes import FakeKB, FakeSession, FakeToolRepo, make_plan

_SENTINEL = object()


def _req(plan, **kw):
    base = dict(message="how many sales orders", plan=plan, app_id="bk")
    base.update(kw)
    return RetrievalRequest(**base)


# ── MetadataRetriever ─────────────────────────────────────────────────────────

async def test_metadata_retriever_emits_entities_and_associations(monkeypatch):
    registry = {"bk": [{
        "app_name": "BK",
        "service_url": "/odata/v4/svc",
        "entities": ["SalesOrder"],
        "entity_fields": {"SalesOrder": ["ID", "BlendID"]},
        "entity_associations": [{"source": "SalesOrder", "target": "Item", "fk_field": "to_Item"}],
        # NOTE: no entity_aliases (live-registered entry) — must not crash.
    }]}
    monkeypatch.setattr(apps_mod, "_service_tool_registry", registry)
    res = await MetadataRetriever().retrieve(_req(make_plan([RetrievalSource.METADATA])))
    kinds = {i.data.get("kind") for i in res.items}
    assert {"service", "entity", "association"}.issubset(kinds)
    entity_item = next(i for i in res.items if i.data.get("kind") == "entity")
    assert entity_item.data["entity"] == "SalesOrder"
    assert entity_item.data["fields"] == ["ID", "BlendID"]


async def test_metadata_retriever_no_app_id_is_empty():
    res = await MetadataRetriever().retrieve(_req(make_plan([RetrievalSource.METADATA]), app_id=None))
    assert res.items == []


# ── ToolRetriever ─────────────────────────────────────────────────────────────

async def test_tool_retriever_returns_tools():
    repo = FakeToolRepo([make_tool("Svc.createOrder", name="createOrder", required=["BlendID"])])
    res = await ToolRetriever(repo=repo).retrieve(_req(make_plan([RetrievalSource.TOOL_REGISTRY]), session=_SENTINEL))
    assert len(res.items) == 1
    assert res.items[0].ref == "Svc.createOrder"
    assert "requiredParameters" in res.items[0].data  # full def preserved (camelCase)


async def test_tool_retriever_no_session_skips_db():
    repo = FakeToolRepo([make_tool("Svc.createOrder")])
    res = await ToolRetriever(repo=repo).retrieve(_req(make_plan([RetrievalSource.TOOL_REGISTRY]), session=None))
    assert res.items == []
    assert repo.calls == 0


# ── VectorRetriever ───────────────────────────────────────────────────────────

async def test_vector_retriever_maps_kb_results():
    kb = FakeKB([{"content": "Orders doc", "score": 0.82, "metadata": {"title": "Orders"}}])
    res = await VectorRetriever(kb_provider=lambda: kb).retrieve(_req(make_plan([RetrievalSource.PGVECTOR])))
    assert len(res.items) == 1
    assert res.items[0].content == "Orders doc"
    assert res.items[0].tier == "semantic"
    assert res.items[0].score == 0.82
    assert kb.vector_store.calls == 1


# ── KeywordRetriever ──────────────────────────────────────────────────────────

async def test_keyword_retriever_fts_rows():
    rows = [{"title": "Orders", "content": "order text", "document_type": "schema",
             "metadata": {"x": 1}, "rank": 0.5}]
    res = await KeywordRetriever().retrieve(_req(make_plan([RetrievalSource.FULL_TEXT_SEARCH]),
                                                 session=FakeSession(rows)))
    assert len(res.items) == 1
    assert res.items[0].content == "order text"
    assert res.items[0].score == 0.5
    assert res.items[0].tier == "exact"


async def test_keyword_retriever_no_session_empty():
    res = await KeywordRetriever().retrieve(_req(make_plan([RetrievalSource.FULL_TEXT_SEARCH]), session=None))
    assert res.items == []


async def test_keyword_retriever_db_error_is_isolated():
    res = await KeywordRetriever().retrieve(_req(make_plan([RetrievalSource.FULL_TEXT_SEARCH]),
                                                 session=FakeSession(raise_exc=RuntimeError("db down"))))
    assert res.items == []
    assert res.error is not None


# ── LiveODataRetriever ────────────────────────────────────────────────────────

async def test_live_odata_retriever_fetches_count_and_rows():
    async def fake_fetch(base, entity_set, headers, top):
        return {"set": entity_set, "rows": [{"ID": 1}], "count": 42}

    plan = make_plan([RetrievalSource.LIVE_ODATA], entity="SalesOrder", requires_live=True)
    req = _req(plan, fiori_context={"service_url": "http://h/odata/v4/svc", "odata_token": "abc"})
    res = await LiveODataRetriever(fetcher=fake_fetch).retrieve(req)
    assert len(res.items) == 1
    assert res.items[0].data["count"] == 42
    assert res.items[0].data["rows"] == [{"ID": 1}]


async def test_live_odata_retriever_tries_set_name_candidates():
    seen = []

    async def fake_fetch(base, entity_set, headers, top):
        seen.append(entity_set)
        return {"set": entity_set, "rows": [], "count": 7} if entity_set.endswith("s") else None

    plan = make_plan([RetrievalSource.LIVE_ODATA], entity="SalesOrder", requires_live=True)
    req = _req(plan, fiori_context={"service_url": "http://h/odata/v4/svc"})
    res = await LiveODataRetriever(fetcher=fake_fetch).retrieve(req)
    assert seen == ["SalesOrder", "SalesOrders"]      # tried type, then plural
    assert res.items[0].data["count"] == 7


async def test_live_odata_retriever_respects_requires_live_false():
    async def fake_fetch(*a, **k):
        raise AssertionError("must not fetch when requires_live_data is False")

    plan = make_plan([RetrievalSource.LIVE_ODATA], entity="SalesOrder", requires_live=False)
    req = _req(plan, fiori_context={"service_url": "http://h/odata/v4/svc"})
    res = await LiveODataRetriever(fetcher=fake_fetch).retrieve(req)
    assert res.items == []


# ── placeholders ──────────────────────────────────────────────────────────────

async def test_placeholders_return_empty():
    plan = make_plan([RetrievalSource.CONVERSATION_MEMORY, RetrievalSource.DOCUMENTATION])
    assert (await MemoryRetriever().retrieve(_req(plan))).items == []
    assert (await DocumentationRetriever().retrieve(_req(plan))).items == []
