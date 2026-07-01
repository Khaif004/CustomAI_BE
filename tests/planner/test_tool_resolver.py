"""Unit tests — ToolResolver (async, faked repository; no DB/LLM)."""
import pytest

from app.services.planner.text_signals import PlanSignals
from app.services.planner.tool_resolver import ToolResolver
from tests.planner.fakes import FakeToolRepository, make_tool

_SESSION = object()  # opaque non-None session sentinel (fake repo ignores it)

PROCESS_TOOLS = {
    "bk": [
        make_tool(
            "ProcessOrderService.createProcessOrder",
            name="createProcessOrder",
            entity="ProcessOrder",
            params=["BlendID", "Quantity"],
            required=["BlendID"],
        ),
        make_tool("OrderService.approveOrder", name="approveOrder", entity="Order"),
    ]
}


def signals(message, app_id="bk", fiori_context=None):
    return PlanSignals.build(message, app_id=app_id, fiori_context=fiori_context)


async def test_matches_tool_and_reports_missing_required_param():
    resolver = ToolResolver(FakeToolRepository(PROCESS_TOOLS))
    res = await resolver.resolve(
        _SESSION, signals("create a process order"), "bk", resolved_entity="ProcessOrder"
    )
    assert res.tool_key == "ProcessOrderService.createProcessOrder"
    assert res.missing_parameters == ["BlendID"]


async def test_required_param_satisfied_when_mentioned():
    resolver = ToolResolver(FakeToolRepository(PROCESS_TOOLS))
    res = await resolver.resolve(
        _SESSION,
        signals("create a process order for blend B-100 with quantity 5"),
        "bk",
        resolved_entity="ProcessOrder",
    )
    assert res.tool_key == "ProcessOrderService.createProcessOrder"
    assert res.missing_parameters == []


async def test_required_param_satisfied_from_fiori_record():
    resolver = ToolResolver(FakeToolRepository(PROCESS_TOOLS))
    res = await resolver.resolve(
        _SESSION,
        signals("create a process order", fiori_context={"entity_data": {"BlendID": "B-1"}}),
        "bk",
        resolved_entity="ProcessOrder",
        fiori_context={"entity_data": {"BlendID": "B-1"}},
    )
    assert res.missing_parameters == []


async def test_verb_entity_binding_selects_correct_tool():
    resolver = ToolResolver(FakeToolRepository(PROCESS_TOOLS))
    res = await resolver.resolve(
        _SESSION, signals("approve the order"), "bk", resolved_entity="Order"
    )
    assert res.tool_key == "OrderService.approveOrder"


async def test_no_app_id_skips_db_call():
    repo = FakeToolRepository(PROCESS_TOOLS)
    resolver = ToolResolver(repo)
    res = await resolver.resolve(_SESSION, signals("create a process order", app_id=None), None)
    assert res.tool_key is None
    assert repo.calls == 0


async def test_none_session_no_crash_and_no_call():
    repo = FakeToolRepository(PROCESS_TOOLS)
    resolver = ToolResolver(repo)
    res = await resolver.resolve(None, signals("create a process order"), "bk")
    assert res.tool_key is None
    assert repo.calls == 0


async def test_no_tools_registered_returns_none():
    resolver = ToolResolver(FakeToolRepository({"bk": []}))
    res = await resolver.resolve(_SESSION, signals("create a process order"), "bk")
    assert res.tool_key is None
