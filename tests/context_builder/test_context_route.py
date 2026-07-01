"""Unit test — POST /api/context/build (fresh app + dependency overrides).

No app.main startup, no real DB/LLM/network. Confirms the
Planner→Orchestrator→Builder wiring + camelCase LLMContext response shape. The
orchestrator is faked to return a known RetrievalContext; the REAL ContextBuilder
turns it into the LLMContext.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import context_routes
from app.auth.security import get_current_user
from app.db.session import get_optional_db
from app.models.planner import Intent, PlannerResult, RetrievalSource
from app.services.context_builder import get_context_builder
from app.services.planner import get_planner_service
from app.services.retrieval import get_retrieval_orchestrator
from tests.context_builder.helpers import rcontext, ritem


class _FakePlanner:
    async def analyze(self, message, *, app_id=None, fiori_context=None, session=None):
        return PlannerResult(
            intent=Intent.DATA_QUERY, confidence=0.9, application=app_id,
            entity="SalesOrder", tool=None,
            retrieval_sources=[RetrievalSource.METADATA, RetrievalSource.PGVECTOR],
            requires_live_data=False, missing_parameters=[],
        )


class _FakeOrchestrator:
    async def retrieve(self, request):
        return rcontext(
            application=request.app_id,
            metadata=[ritem(RetrievalSource.METADATA, ref="entity:SalesOrder", title="SalesOrder",
                            data={"kind": "entity", "entity": "SalesOrder", "fields": ["ID"]})],
            semantic_documents=[ritem(RetrievalSource.PGVECTOR, tier="semantic", ref="s1",
                                      content="semantic doc about orders", score=0.7)],
        )


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(context_routes.router)
    app.dependency_overrides[get_planner_service] = lambda: _FakePlanner()
    app.dependency_overrides[get_retrieval_orchestrator] = lambda: _FakeOrchestrator()
    app.dependency_overrides[get_context_builder] = get_context_builder  # real builder
    app.dependency_overrides[get_optional_db] = lambda: None
    app.dependency_overrides[get_current_user] = lambda: {"sub": "tester"}
    return app


def test_context_build_endpoint():
    client = TestClient(_make_app())
    resp = client.post("/api/context/build", json={"message": "show me sales orders", "app_id": "bk"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # camelCase sections present
    for key in ("systemInstructions", "applicationMetadata", "liveBusinessData", "toolMetadata",
                "documentation", "conversationContext", "currentUIContext", "semanticKnowledge",
                "statistics"):
        assert key in body, key
    assert body["application"] == "bk"
    assert len(body["applicationMetadata"]) == 1
    assert body["applicationMetadata"][0]["source"] == "Metadata"
    assert body["applicationMetadata"][0]["retriever"] == "MetadataRetriever"
    assert len(body["semanticKnowledge"]) == 1
    stats = body["statistics"]
    assert stats["tokenEstimate"] > 0
    assert "MetadataRetriever" in stats["retrieversUsed"]
    assert "compressionRatio" in stats and "duplicateCount" in stats and "documentsDiscarded" in stats
