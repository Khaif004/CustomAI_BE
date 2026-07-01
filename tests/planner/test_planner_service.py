"""Unit tests — PlannerService end-to-end (faked ports; no DB/LLM)."""
import pytest

from app.models.planner import Intent, RetrievalSource
from app.services.planner.entity_resolver import EntityResolver
from app.services.planner.intent_classifier import IntentClassifier
from app.services.planner.planner_service import PlannerService
from app.services.planner.tool_resolver import ToolResolver
from tests.planner.fakes import FakeEntityRegistry, FakeToolRepository, make_tool

_SESSION = object()

ENTITIES = {"bk": ["ProcessOrder", "SalesOrder", "Material", "Supplier"]}
TOOLS = {
    "bk": [
        make_tool(
            "ProcessOrderService.createProcessOrder",
            name="createProcessOrder",
            entity="ProcessOrder",
            params=["BlendID", "Quantity"],
            required=["BlendID"],
        )
    ]
}


def build_service():
    return PlannerService(
        intent_classifier=IntentClassifier(),
        entity_resolver=EntityResolver(FakeEntityRegistry(ENTITIES)),
        tool_resolver=ToolResolver(FakeToolRepository(TOOLS)),
    )


async def test_tool_execution_plan_matches_spec_example():
    svc = build_service()
    plan = await svc.analyze("create a process order", app_id="bk", session=_SESSION)
    assert plan.intent is Intent.TOOL_EXECUTION
    assert plan.application == "bk"
    assert plan.entity == "ProcessOrder"
    assert plan.tool == "ProcessOrderService.createProcessOrder"
    assert plan.missing_parameters == ["BlendID"]
    assert plan.requires_live_data is False
    assert RetrievalSource.TOOL_REGISTRY in plan.retrieval_sources
    assert RetrievalSource.METADATA in plan.retrieval_sources
    assert RetrievalSource.LIVE_ODATA not in plan.retrieval_sources
    assert 0.0 <= plan.confidence <= 0.99


async def test_data_query_requires_live_data_and_no_db_call():
    repo = FakeToolRepository(TOOLS)
    svc = PlannerService(IntentClassifier(), EntityResolver(FakeEntityRegistry(ENTITIES)), ToolResolver(repo))
    plan = await svc.analyze("show me all sales orders", app_id="bk", session=_SESSION)
    assert plan.intent is Intent.DATA_QUERY
    assert plan.entity == "SalesOrder"
    assert plan.requires_live_data is True
    assert plan.retrieval_sources[0] is RetrievalSource.LIVE_ODATA
    assert plan.tool is None
    assert plan.missing_parameters == []
    # A read intent must NOT consult the Tool Registry (no needless DB hit).
    assert repo.calls == 0


async def test_knowledge_no_app_no_live_data():
    svc = build_service()
    plan = await svc.analyze("what is a process order", app_id=None, session=_SESSION)
    assert plan.intent is Intent.KNOWLEDGE
    assert plan.requires_live_data is False
    assert plan.application is None
    assert RetrievalSource.PGVECTOR in plan.retrieval_sources


async def test_graceful_without_db_session():
    # Neon down → session is None. Intent still classified; tool stays unresolved.
    svc = build_service()
    plan = await svc.analyze("create a process order", app_id="bk", session=None)
    assert plan.intent is Intent.TOOL_EXECUTION
    assert plan.tool is None              # could not consult the registry
    assert plan.missing_parameters == []


async def test_data_query_without_entity_does_not_require_live_data():
    # entity registry empty for this app → no target entity → no live fetch.
    svc = PlannerService(
        IntentClassifier(),
        EntityResolver(FakeEntityRegistry({"bk": []})),
        ToolResolver(FakeToolRepository({"bk": []})),
    )
    plan = await svc.analyze("show me all widgets", app_id="bk", session=_SESSION)
    assert plan.intent is Intent.DATA_QUERY
    assert plan.entity is None
    assert plan.requires_live_data is False


async def test_result_serializes_camelCase():
    svc = build_service()
    plan = await svc.analyze("create a process order", app_id="bk", session=_SESSION)
    dumped = plan.model_dump(by_alias=True)
    assert set(["intent", "confidence", "application", "entity", "tool",
                "retrievalSources", "requiresLiveData", "missingParameters"]).issubset(dumped.keys())


async def test_planner_never_imports_llm_client():
    # Enforce the "never calls the LLM" constraint. We assert it at the SOURCE level
    # (the planner package imports no LLM/HTTP client) rather than via global
    # sys.modules — the latter is polluted by sibling test modules that legitimately
    # import the chat agents (which import langchain_openai). Source scan is both
    # pollution-proof and a stronger guarantee.
    import pathlib
    import app.services.planner as planner_pkg

    # The planner still runs to completion with no network / DB session.
    svc = build_service()
    await svc.analyze("create a process order", app_id="bk", session=_SESSION)

    banned = ("import openai", "from openai", "import anthropic", "from anthropic",
              "langchain_openai", "langchain_anthropic")
    pkg_dir = pathlib.Path(planner_pkg.__file__).parent
    for py in pkg_dir.rglob("*.py"):
        src = py.read_text(encoding="utf-8")
        for token in banned:
            assert token not in src, f"planner module {py.name} references an LLM/HTTP client: {token!r}"
