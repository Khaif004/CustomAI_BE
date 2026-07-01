"""Unit tests — ODataExecutor response normalisation (no real HTTP calls).

Each test constructs a fake aiohttp response and exercises _parse_response
directly so network / retry logic is bypassed.
"""
import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.action_execution.exceptions import ODataExecutionError
from app.services.action_execution.odata_executor import ODataExecutor

_exec = ODataExecutor()
_URL = "https://cap/odata/v4/Service/Action"


def _fake_resp(
    status: int,
    json_data: Optional[Any] = None,
    text_data: str = "",
    content_type: str = "application/json",
) -> MagicMock:
    """Build a fake aiohttp ClientResponse-like object."""
    resp = MagicMock()
    resp.status = status
    resp.headers = {"Content-Type": content_type}
    resp.json = AsyncMock(return_value=json_data)
    resp.text = AsyncMock(return_value=text_data if json_data is None else json.dumps(json_data))
    return resp


# ── 204 No Content ────────────────────────────────────────────────────────────

async def test_204_returns_none_result():
    resp = _fake_resp(204)
    raw = await ODataExecutor._parse_response(resp, _URL, 5.0)
    assert raw.http_status == 204
    assert raw.result is None
    assert "successfully" in raw.messages[0].lower()


# ── 200 with OData value array ────────────────────────────────────────────────

async def test_200_unwraps_odata_value_array():
    data = {"value": [{"ID": 1}, {"ID": 2}], "@odata.count": 2}
    resp = _fake_resp(200, json_data=data)
    raw = await ODataExecutor._parse_response(resp, _URL, 10.0)
    assert raw.http_status == 200
    assert raw.result == [{"ID": 1}, {"ID": 2}]


async def test_200_no_value_key_returns_full_dict():
    data = {"OrderID": "abc", "status": "Released"}
    resp = _fake_resp(200, json_data=data)
    raw = await ODataExecutor._parse_response(resp, _URL, 10.0)
    assert raw.result == data


async def test_200_strips_odata_context_metadata():
    data = {
        "@odata.context": "$metadata#Entity",
        "@odata.metadataEtag": "W/\"abc\"",
        "value": [{"ID": 1}],
    }
    resp = _fake_resp(200, json_data=data)
    raw = await ODataExecutor._parse_response(resp, _URL, 10.0)
    # result is the unwrapped value list; context key stripped from it if dict
    assert isinstance(raw.result, list)


async def test_200_captures_sap_messages():
    data = {
        "value": {"result": "ok"},
        "@sap.messages": [{"message": "Goods receipt posted.", "severity": "Info"}],
    }
    resp = _fake_resp(200, json_data=data)
    raw = await ODataExecutor._parse_response(resp, _URL, 10.0)
    assert any("Goods receipt" in m for m in raw.messages)


# ── 4xx / 5xx errors ─────────────────────────────────────────────────────────

async def test_400_raises_odata_execution_error():
    err_body = {"error": {"code": "BAD_REQUEST", "message": "Invalid OrderID format."}}
    resp = _fake_resp(400, json_data=err_body)
    with pytest.raises(ODataExecutionError) as exc_info:
        await ODataExecutor._parse_response(resp, _URL, 5.0)
    assert exc_info.value.status_code == 400
    assert "Invalid OrderID" in exc_info.value.detail


async def test_401_raises_odata_execution_error():
    err_body = {"error": {"message": "Unauthorized"}}
    resp = _fake_resp(401, json_data=err_body)
    with pytest.raises(ODataExecutionError) as exc_info:
        await ODataExecutor._parse_response(resp, _URL, 5.0)
    assert exc_info.value.status_code == 401


async def test_403_raises_odata_execution_error():
    resp = _fake_resp(403, text_data="Forbidden", content_type="text/plain")
    resp.json = AsyncMock(side_effect=Exception("not json"))
    resp.text = AsyncMock(return_value="Forbidden")
    with pytest.raises(ODataExecutionError) as exc_info:
        await ODataExecutor._parse_response(resp, _URL, 5.0)
    assert exc_info.value.status_code == 403


async def test_404_raises_odata_execution_error():
    err_body = {"error": {"message": "Entity not found."}}
    resp = _fake_resp(404, json_data=err_body)
    with pytest.raises(ODataExecutionError) as exc_info:
        await ODataExecutor._parse_response(resp, _URL, 5.0)
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


async def test_500_raises_odata_execution_error():
    resp = _fake_resp(500, text_data="Internal Server Error", content_type="text/plain")
    resp.json = AsyncMock(side_effect=Exception("not json"))
    resp.text = AsyncMock(return_value="Internal Server Error")
    with pytest.raises(ODataExecutionError) as exc_info:
        await ODataExecutor._parse_response(resp, _URL, 5.0)
    assert exc_info.value.status_code == 500


# ── Plain text response ───────────────────────────────────────────────────────

async def test_plain_text_response_returned_as_string():
    resp = _fake_resp(200, text_data="42", content_type="text/plain")
    resp.json = AsyncMock(side_effect=Exception("not json"))
    resp.text = AsyncMock(return_value="42")
    raw = await ODataExecutor._parse_response(resp, _URL, 5.0)
    assert raw.result == "42"


# ── Duration and URL preserved ────────────────────────────────────────────────

async def test_duration_and_url_preserved():
    resp = _fake_resp(200, json_data={"value": []})
    raw = await ODataExecutor._parse_response(resp, _URL, 123.45)
    assert raw.duration_ms == 123.45
    assert raw.raw_url == _URL
