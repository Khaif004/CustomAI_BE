"""Unit test — POST /api/retrieval/context (fresh app + dependency overrides).

No app.main startup, no real DB/LLM/network. Confirms the Planner→Orchestrator
wiring + camelCase RetrievalContext response shape.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import retrieval_routes
from app.auth.security import get_current_user
from app.db.session import get_optional_db
from app.models.planner import RetrievalSource
from app.services.planner import get_planner_service
from app.services.retrieval import get_retrieval_orchestrator
from app.services.retrieval.merger import ResultMerger
from app.services.retrieval.models import Section
from app.services.retrieval.orchestrator import RetrievalOrchestrator
from tests.retrieval.fakes import FakePlanner, FakeRetriever, item, make_plan


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(retrieval_routes.router)

    plan = make_plan([RetrievalSource.METADATA, RetrievalSource.PGVECTOR],
                     entity="SalesOrder", application="bk")
    meta = FakeRetriever(RetrievalSource.METADATA, Section.METADATA,
                         items=[item(RetrievalSource.METADATA, "e:SalesOrder", title="SalesOrder")])
    vec = FakeRetriever(RetrievalSource.PGVECTOR, Section.SEMANTIC_DOCUMENTS,
                        items=[item(RetrievalSource.PGVECTOR, "d1", tier="semantic", score=0.7, content="doc")])
    orch = RetrievalOrchestrator([meta, vec], ResultMerger())

    app.dependency_overrides[get_planner_service] = lambda: FakePlanner(plan)
    app.dependency_overrides[get_retrieval_orchestrator] = lambda: orch
    app.dependency_overrides[get_optional_db] = lambda: None
    app.dependency_overrides[get_current_user] = lambda: {"sub": "tester"}
    return app


def test_retrieval_context_endpoint():
    client = TestClient(_make_app())
    resp = client.post("/api/retrieval/context", json={"message": "show me sales orders", "app_id": "bk"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # camelCase sections present
    assert "semanticDocuments" in body and "keywordMatches" in body and "liveData" in body
    assert body["application"] == "bk"
    assert len(body["metadata"]) == 1
    assert body["metadata"][0]["source"] == "Metadata"
    assert len(body["semanticDocuments"]) == 1
    assert body["semanticDocuments"][0]["source"] == "Pgvector"
    assert set(body["sourcesRun"]) == {"Metadata", "Pgvector"}
