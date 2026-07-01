"""Unit test — the additive POST /api/planner/analyze route.

Builds a throwaway FastAPI app with ONLY the planner router (so no app.main
startup, no real DB/LLM) and overrides the three dependencies. Confirms the
wiring + camelCase response shape without touching chat routes.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import planner_routes
from app.auth.security import get_current_user
from app.db.session import get_optional_db
from app.services.planner import get_planner_service
from app.services.planner.entity_resolver import EntityResolver
from app.services.planner.intent_classifier import IntentClassifier
from app.services.planner.planner_service import PlannerService
from app.services.planner.tool_resolver import ToolResolver
from tests.planner.fakes import FakeEntityRegistry, FakeToolRepository, make_tool

_SESSION = object()
ENTITIES = {"bk": ["ProcessOrder"]}
TOOLS = {
    "bk": [
        make_tool(
            "ProcessOrderService.createProcessOrder",
            name="createProcessOrder",
            entity="ProcessOrder",
            params=["BlendID"],
            required=["BlendID"],
        )
    ]
}


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(planner_routes.router)
    fake_planner = PlannerService(
        IntentClassifier(),
        EntityResolver(FakeEntityRegistry(ENTITIES)),
        ToolResolver(FakeToolRepository(TOOLS)),
    )
    app.dependency_overrides[get_planner_service] = lambda: fake_planner
    app.dependency_overrides[get_optional_db] = lambda: _SESSION
    app.dependency_overrides[get_current_user] = lambda: {"sub": "tester"}
    return app


def test_analyze_endpoint_returns_camelcase_plan():
    client = TestClient(_make_app())
    resp = client.post(
        "/api/planner/analyze",
        json={"message": "create a process order", "app_id": "bk"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["intent"] == "TOOL_EXECUTION"
    assert body["application"] == "bk"
    assert body["entity"] == "ProcessOrder"
    assert body["tool"] == "ProcessOrderService.createProcessOrder"
    assert body["missingParameters"] == ["BlendID"]
    assert body["requiresLiveData"] is False
    assert "ToolRegistry" in body["retrievalSources"]


def test_analyze_endpoint_data_query():
    client = TestClient(_make_app())
    resp = client.post(
        "/api/planner/analyze",
        json={"message": "show me all process orders", "app_id": "bk"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["intent"] == "DATA_QUERY"
    assert body["requiresLiveData"] is True
    assert body["retrievalSources"][0] == "LiveOData"
