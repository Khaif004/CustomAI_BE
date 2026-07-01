"""Unit tests — ActionExecutionService orchestration (no real HTTP, no real DB)."""
import pytest

from app.models.tool_catalog import Authorization, ToolBinding, ToolType
from app.services.action_execution.exceptions import ODataExecutionError
from app.services.action_execution.executor import ActionExecutionService
from app.services.action_execution.models import ToolExecutionStatus
from app.services.action_execution.odata_executor import ODataRawResponse
from tests.action_execution.conftest import (
    FakeODataExecutor,
    make_param,
    make_request,
    make_tool,
)

_SERVICE_URL = "https://cap.cfapps.eu10.hana.ondemand.com/odata/v4/ProcessOrderService"


def _make_service(tool, odata_executor=None, monkeypatch=None, app_id="process-orders"):
    """Return an ActionExecutionService pre-wired with a fake OData executor
    and a patched _load_tool / _resolve_service_url so no DB or HTTP is needed."""
    ex = odata_executor or FakeODataExecutor()
    svc = ActionExecutionService(odata_executor=ex)

    async def _fake_load(self, aid, tk, session):
        return tool

    # staticmethod replaced by monkeypatch becomes a regular method — needs self
    def _fake_resolve(self, aid, t):  # noqa: N805
        return _SERVICE_URL

    if monkeypatch:
        monkeypatch.setattr(ActionExecutionService, "_load_tool", _fake_load)
        monkeypatch.setattr(ActionExecutionService, "_resolve_service_url", _fake_resolve)

    return svc


# ── Happy path ────────────────────────────────────────────────────────────────

async def test_success_result_returned(monkeypatch):
    tool = make_tool()
    svc = _make_service(tool, monkeypatch=monkeypatch)
    result = await svc.execute(make_request())
    assert result.success is True
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.http_status_code == 200
    assert result.result is not None
    assert result.execution_time_ms > 0


async def test_204_no_content_success(monkeypatch):
    fake_ex = FakeODataExecutor(response=ODataRawResponse(
        http_status=204, result=None,
        messages=["Action executed successfully."],
        raw_url=_SERVICE_URL, duration_ms=5.0,
    ))
    tool = make_tool()
    svc = _make_service(tool, odata_executor=fake_ex, monkeypatch=monkeypatch)
    result = await svc.execute(make_request())
    assert result.success is True
    assert result.http_status_code == 204
    assert result.result is None


# ── Tool not found ────────────────────────────────────────────────────────────

async def test_tool_not_found_returns_not_found_status(monkeypatch):
    svc = ActionExecutionService()

    async def _not_found(self, aid, tk, session):
        from app.services.action_execution.exceptions import ToolNotFoundError
        raise ToolNotFoundError(aid, tk)

    monkeypatch.setattr(ActionExecutionService, "_load_tool", _not_found)
    result = await svc.execute(make_request())
    assert result.status == ToolExecutionStatus.NOT_FOUND
    assert result.success is False
    assert result.error is not None
    assert result.error.code == "TOOL_NOT_FOUND"


# ── Authorization ─────────────────────────────────────────────────────────────

async def test_authorization_denied_when_no_matching_role(monkeypatch):
    tool = make_tool(authorization=Authorization(
        required_roles=["ProcessOrder.Release"],
        restrictions=[],
    ))
    svc = _make_service(tool, monkeypatch=monkeypatch)
    req = make_request(user_roles=[])   # user has no roles
    result = await svc.execute(req)
    assert result.status == ToolExecutionStatus.AUTH_ERROR
    assert result.success is False
    assert result.error.code == "AUTHORIZATION_DENIED"


async def test_authorization_passes_with_matching_role(monkeypatch):
    tool = make_tool(authorization=Authorization(
        required_roles=["ProcessOrder.Release"],
        restrictions=[],
    ))
    svc = _make_service(tool, monkeypatch=monkeypatch)
    req = make_request(user_roles=["ProcessOrder.Release"])
    result = await svc.execute(req)
    assert result.success is True


async def test_authorization_passes_any_of_required_roles(monkeypatch):
    tool = make_tool(authorization=Authorization(
        required_roles=["Admin", "ProcessOrder.Release"],
        restrictions=[],
    ))
    svc = _make_service(tool, monkeypatch=monkeypatch)
    req = make_request(user_roles=["ProcessOrder.Release"])
    result = await svc.execute(req)
    assert result.success is True


async def test_no_required_roles_always_passes(monkeypatch):
    tool = make_tool(authorization=Authorization(required_roles=[], restrictions=[]))
    svc = _make_service(tool, monkeypatch=monkeypatch)
    result = await svc.execute(make_request(user_roles=[]))
    assert result.success is True


async def test_no_authorization_object_always_passes(monkeypatch):
    tool = make_tool(authorization=None)
    svc = _make_service(tool, monkeypatch=monkeypatch)
    result = await svc.execute(make_request(user_roles=[]))
    assert result.success is True


# ── Parameter validation ──────────────────────────────────────────────────────

async def test_missing_required_param_returns_validation_error(monkeypatch):
    tool = make_tool(
        parameters=[make_param("OrderID", "UUID", required=True)],
        required_parameters=["OrderID"],
    )
    svc = _make_service(tool, monkeypatch=monkeypatch)
    result = await svc.execute(make_request(parameters={}))
    assert result.status == ToolExecutionStatus.VALIDATION_ERROR
    assert result.success is False
    assert "OrderID" in (result.error.detail or "")


async def test_valid_params_accepted(monkeypatch):
    tool = make_tool(
        parameters=[make_param("OrderID", "UUID", required=True)],
        required_parameters=["OrderID"],
    )
    svc = _make_service(tool, monkeypatch=monkeypatch)
    result = await svc.execute(
        make_request(parameters={"OrderID": "550e8400-e29b-41d4-a716-446655440000"})
    )
    assert result.success is True


# ── OData errors ──────────────────────────────────────────────────────────────

async def test_odata_400_returns_failed_status(monkeypatch):
    fake_ex = FakeODataExecutor(exc=ODataExecutionError(400, "Invalid key"))
    tool = make_tool()
    svc = _make_service(tool, odata_executor=fake_ex, monkeypatch=monkeypatch)
    result = await svc.execute(make_request())
    assert result.status == ToolExecutionStatus.FAILED
    assert result.success is False
    assert result.error.code == "ODATA_ERROR"


async def test_timeout_returns_timeout_status(monkeypatch):
    import asyncio
    fake_ex = FakeODataExecutor(exc=asyncio.TimeoutError())
    tool = make_tool()
    svc = _make_service(tool, odata_executor=fake_ex, monkeypatch=monkeypatch)
    result = await svc.execute(make_request())
    assert result.status == ToolExecutionStatus.TIMEOUT
    assert result.success is False


async def test_unexpected_exception_returns_failed(monkeypatch):
    fake_ex = FakeODataExecutor(exc=RuntimeError("unexpected boom"))
    tool = make_tool()
    svc = _make_service(tool, odata_executor=fake_ex, monkeypatch=monkeypatch)
    result = await svc.execute(make_request())
    assert result.status == ToolExecutionStatus.FAILED
    assert result.error.code == "UNEXPECTED_ERROR"


# ── Confirmation flag ─────────────────────────────────────────────────────────

async def test_requires_confirmation_true_for_release_action(monkeypatch):
    tool = make_tool(tool_key="ReleaseProcessOrder", tool_type=ToolType.ACTION)
    svc = _make_service(tool, monkeypatch=monkeypatch)
    result = await svc.execute(make_request())
    assert result.requires_confirmation is True


async def test_requires_confirmation_false_for_function(monkeypatch):
    tool = make_tool(
        tool_key="GetOrderCount",
        tool_type=ToolType.FUNCTION,
        http_method="GET",
    )
    fake_ex = FakeODataExecutor()
    svc = _make_service(tool, odata_executor=fake_ex, monkeypatch=monkeypatch)
    result = await svc.execute(make_request(tool_key="GetOrderCount"))
    assert result.requires_confirmation is False
