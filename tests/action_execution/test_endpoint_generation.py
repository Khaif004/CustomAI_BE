"""Unit tests — ODataExecutor URL building (no HTTP calls)."""
import pytest

from app.models.tool_catalog import ToolBinding, ToolType
from app.services.action_execution.exceptions import (
    ConfigurationError,
    EndpointResolutionError,
)
from app.services.action_execution.odata_executor import ODataExecutor, build_absolute_url
from tests.action_execution.conftest import make_tool

_BASE = "https://cap.cfapps.eu10.hana.ondemand.com/odata/v4/ProcessOrderService"
_exec = ODataExecutor()


# ── Unbound action ────────────────────────────────────────────────────────────

def test_unbound_action_simple_endpoint():
    tool = make_tool(
        binding=ToolBinding.UNBOUND,
        http_endpoint="ReleaseProcessOrder",
        tool_type=ToolType.ACTION,
    )
    url = _exec._build_url(tool, {}, None, _BASE)
    assert url == f"{_BASE}/ReleaseProcessOrder"


def test_unbound_action_trailing_slash_normalised():
    tool = make_tool(binding=ToolBinding.UNBOUND, http_endpoint="/ReleaseProcessOrder")
    url = _exec._build_url(tool, {}, None, _BASE + "/")
    assert url == f"{_BASE}/ReleaseProcessOrder"


# ── Bound action ──────────────────────────────────────────────────────────────

def test_bound_action_angle_bracket_key_replaced():
    tool = make_tool(
        binding=ToolBinding.BOUND,
        http_endpoint="ProcessOrders(<key>)/com.sap.cap/ReleaseOrder",
    )
    url = _exec._build_url(tool, {}, "abc-uuid-123", _BASE)
    assert url == f"{_BASE}/ProcessOrders(abc-uuid-123)/com.sap.cap/ReleaseOrder"


def test_bound_action_curly_brace_key_replaced():
    tool = make_tool(
        binding=ToolBinding.BOUND,
        http_endpoint="ProcessOrders({key})/ReleaseOrder",
    )
    url = _exec._build_url(tool, {}, "my-key", _BASE)
    assert url == f"{_BASE}/ProcessOrders(my-key)/ReleaseOrder"


def test_bound_action_uppercase_key_replaced():
    tool = make_tool(
        binding=ToolBinding.BOUND,
        http_endpoint="ProcessOrders({KEY})/ReleaseOrder",
    )
    url = _exec._build_url(tool, {}, "my-key", _BASE)
    assert url == f"{_BASE}/ProcessOrders(my-key)/ReleaseOrder"


def test_bound_action_missing_key_raises():
    tool = make_tool(
        binding=ToolBinding.BOUND,
        http_endpoint="ProcessOrders(<key>)/ReleaseOrder",
    )
    with pytest.raises(EndpointResolutionError, match="entity_key"):
        _exec._build_url(tool, {}, None, _BASE)


# ── Function ──────────────────────────────────────────────────────────────────

def test_unbound_function_params_appended_as_query_string():
    tool = make_tool(
        tool_type=ToolType.FUNCTION,
        binding=ToolBinding.UNBOUND,
        http_method="GET",
        http_endpoint="GetOrderCount",
    )
    url = _exec._build_url(tool, {"plant": "1000", "material": "MAT-01"}, None, _BASE)
    assert url.startswith(f"{_BASE}/GetOrderCount?")
    assert "plant=1000" in url
    assert "material=MAT-01" in url


def test_function_no_params_url_unchanged():
    tool = make_tool(
        tool_type=ToolType.FUNCTION,
        binding=ToolBinding.UNBOUND,
        http_method="GET",
        http_endpoint="GetServerTime",
    )
    url = _exec._build_url(tool, {}, None, _BASE)
    assert url == f"{_BASE}/GetServerTime"


def test_function_with_embedded_params_not_doubled():
    # Endpoint already has inline params like FunctionName(param=value) —
    # must NOT append duplicates.
    tool = make_tool(
        tool_type=ToolType.FUNCTION,
        binding=ToolBinding.UNBOUND,
        http_method="GET",
        http_endpoint="GetOrderCount(plant='1000')",
    )
    url = _exec._build_url(tool, {"plant": "1000"}, None, _BASE)
    assert url.count("plant") == 1


def test_bound_function_key_replaced_and_params_appended():
    tool = make_tool(
        tool_type=ToolType.FUNCTION,
        binding=ToolBinding.BOUND,
        http_method="GET",
        http_endpoint="Orders(<key>)/GetDetails",
    )
    url = _exec._build_url(tool, {"expand": "items"}, "ord-uuid", _BASE)
    assert "ord-uuid" in url
    assert "expand=items" in url


# ── Configuration errors ──────────────────────────────────────────────────────

def test_empty_service_base_url_raises():
    tool = make_tool(binding=ToolBinding.UNBOUND, http_endpoint="SomeAction")
    with pytest.raises(ConfigurationError):
        _exec._build_url(tool, {}, None, "")


def test_missing_http_endpoint_raises():
    tool = make_tool(binding=ToolBinding.UNBOUND, http_endpoint="")
    tool.http_endpoint = None  # type: ignore[assignment]
    with pytest.raises(EndpointResolutionError):
        _exec._build_url(tool, {}, None, _BASE)


# ── build_absolute_url unit tests ─────────────────────────────────────────────

class TestBuildAbsoluteUrl:
    """Tests for the module-level build_absolute_url() helper.

    This helper is the single authoritative place for URL assembly; ODataExecutor
    delegates to it.  Tests cover every scenario listed in the redesign spec.
    """

    # ── Happy-path: normal joins ───────────────────────────────────────────────

    def test_simple_absolute_base_and_path(self):
        url = build_absolute_url("https://host.example.com", "/odata/v4/Svc/Entity")
        assert url == "https://host.example.com/odata/v4/Svc/Entity"

    def test_trailing_slash_on_base_is_normalised(self):
        url = build_absolute_url("https://host.example.com/", "/odata/v4/Svc/Entity")
        assert url == "https://host.example.com/odata/v4/Svc/Entity"

    def test_leading_slash_on_endpoint_is_accepted(self):
        url = build_absolute_url("https://host.example.com", "/Entity/Action")
        assert url == "https://host.example.com/Entity/Action"

    def test_no_leading_slash_on_endpoint_is_normalised(self):
        url = build_absolute_url("https://host.example.com", "Entity/Action")
        assert url == "https://host.example.com/Entity/Action"

    def test_base_with_app_path_joined_correctly(self):
        # Base has a non-OData app path that doesn't match the endpoint prefix.
        url = build_absolute_url(
            "https://approuter.host/myapp",
            "/odata/v4/FertilizerBlendService/FertilizerBlend",
        )
        assert url == "https://approuter.host/myapp/odata/v4/FertilizerBlendService/FertilizerBlend"

    # ── Duplicate-prefix collapsing ───────────────────────────────────────────

    def test_duplicate_odata_prefix_collapsed(self):
        # Real duplicate: service_base_url path component and http_endpoint share
        # the same prefix — e.g. the registry stored the full OData service path
        # AND the endpoint also includes it.
        url = build_absolute_url(
            "https://approuter.host/odata/v4/fertilizer-blend",
            "/odata/v4/fertilizer-blend/FertilizerBlend(1856)/FertilizerBlendService.RefreshProcessOrder",
        )
        assert url == (
            "https://approuter.host"
            "/odata/v4/fertilizer-blend"
            "/FertilizerBlend(1856)"
            "/FertilizerBlendService.RefreshProcessOrder"
        )

    def test_duplicate_prefix_collapse_case_insensitive(self):
        # Prefix comparison must be case-insensitive so mixed-case registries work.
        url = build_absolute_url(
            "https://host/OData/V4/MySvc",
            "/odata/v4/MySvc/Entity/Action",
        )
        assert url == "https://host/odata/v4/MySvc/Entity/Action"

    def test_non_matching_base_path_not_collapsed(self):
        # Base path (/app) is NOT a prefix of the endpoint (/odata/v4/svc/…),
        # so we do a normal join — no collapse.
        url = build_absolute_url(
            "https://host/app",
            "/odata/v4/svc/Entity",
        )
        assert url == "https://host/app/odata/v4/svc/Entity"

    def test_duplicate_prefix_exact_match_collapsed(self):
        # Edge case: base_path == endpoint exactly (e.g. listing the service root).
        url = build_absolute_url(
            "https://host/odata/v4/svc",
            "/odata/v4/svc",
        )
        assert url == "https://host/odata/v4/svc"

    # ── Error cases: relative / missing-scheme URLs ───────────────────────────

    def test_relative_service_base_url_raises_endpoint_resolution_error(self):
        # The primary fix: a relative service_base_url must raise a typed error,
        # never surface as an UNEXPECTED_ERROR with the URL as the message.
        with pytest.raises(EndpointResolutionError, match="absolute URL"):
            build_absolute_url(
                "/odata/v4/assign-material-to-formula",
                "/odata/v4/fertilizer-blend/FertilizerBlend(1)/Action",
            )

    def test_no_scheme_raises_endpoint_resolution_error(self):
        with pytest.raises(EndpointResolutionError, match="absolute URL"):
            build_absolute_url("host.example.com", "/Entity/Action")

    def test_empty_base_raises_configuration_error(self):
        with pytest.raises(ConfigurationError):
            build_absolute_url("", "/Entity/Action")

    def test_whitespace_only_base_raises_endpoint_resolution_error(self):
        # "  " stripped → "" → urlparse gives no scheme/host.
        with pytest.raises((ConfigurationError, EndpointResolutionError)):
            build_absolute_url("  ", "/Entity/Action")

    # ── Scenario: service URL ending with "/" ─────────────────────────────────

    def test_service_url_ending_with_slash(self):
        url = build_absolute_url(
            "https://cap.eu10.hana.ondemand.com/odata/v4/ProcessOrderService/",
            "ReleaseProcessOrder",
        )
        assert url == (
            "https://cap.eu10.hana.ondemand.com"
            "/odata/v4/ProcessOrderService"
            "/ReleaseProcessOrder"
        )

    # ── Registry-style scenarios matching real tool data ──────────────────────

    def test_unbound_action_registry_style(self):
        # Absolute base (correct registry entry) + endpoint without prefix
        url = build_absolute_url(
            "https://cap.cfapps.eu10.hana.ondemand.com/odata/v4/ProcessOrderService",
            "ReleaseProcessOrder",
        )
        assert url == (
            "https://cap.cfapps.eu10.hana.ondemand.com"
            "/odata/v4/ProcessOrderService"
            "/ReleaseProcessOrder"
        )

    def test_unbound_function_registry_style(self):
        url = build_absolute_url(
            "https://cap.cfapps.eu10.hana.ondemand.com/odata/v4/ProcessOrderService",
            "GetProcessOrderCount",
        )
        assert url == (
            "https://cap.cfapps.eu10.hana.ondemand.com"
            "/odata/v4/ProcessOrderService"
            "/GetProcessOrderCount"
        )

    def test_bound_action_registry_style(self):
        # Endpoint template already has entity key substituted by _build_url.
        url = build_absolute_url(
            "https://cap.cfapps.eu10.hana.ondemand.com/odata/v4/ProcessOrderService",
            "ProcessOrders(abc-123)/com.sap.cap.ReleaseOrder",
        )
        assert url == (
            "https://cap.cfapps.eu10.hana.ondemand.com"
            "/odata/v4/ProcessOrderService"
            "/ProcessOrders(abc-123)/com.sap.cap.ReleaseOrder"
        )

    def test_bound_function_registry_style(self):
        url = build_absolute_url(
            "https://cap.cfapps.eu10.hana.ondemand.com/odata/v4/FertilizerBlendService",
            "FertilizerBlend(1856)/FertilizerBlendService.RefreshProcessOrder",
        )
        assert url == (
            "https://cap.cfapps.eu10.hana.ondemand.com"
            "/odata/v4/FertilizerBlendService"
            "/FertilizerBlend(1856)/FertilizerBlendService.RefreshProcessOrder"
        )

    def test_stutsman_relative_base_url_raises_with_diagnostic(self):
        # Reproduces the exact UNEXPECTED_ERROR from production.
        # service_base_url = "/odata/v4/assign-material-to-formula" (no host)
        # http_endpoint    = "/odata/v4/fertilizer-blend/FertilizerBlend(1856)/..."
        with pytest.raises(EndpointResolutionError) as exc_info:
            build_absolute_url(
                "/odata/v4/assign-material-to-formula",
                "/odata/v4/fertilizer-blend/FertilizerBlend(1856)/FertilizerBlendService.RefreshProcessOrder",
            )
        msg = str(exc_info.value)
        # Error message should include diagnostic details, not just be the raw URL
        assert "Service base URL" in msg or "absolute URL" in msg

    # ── build_absolute_url called from _build_url ─────────────────────────────
    # (These go through the full ODataExecutor._build_url path, covering the
    #  integration between entity-key substitution and build_absolute_url.)

    def test_executor_bound_action_stutsman_correct_base(self):
        """When service_base_url is corrected to absolute, URL is correct."""
        tool = make_tool(
            binding=ToolBinding.BOUND,
            tool_type=ToolType.ACTION,
            http_endpoint="/odata/v4/fertilizer-blend/FertilizerBlend(<key>)/FertilizerBlendService.RefreshProcessOrder",
        )
        url = _exec._build_url(
            tool,
            {},
            "1856",
            "https://approuter.host/odata/v4/assign-material-to-formula",
        )
        # Duplicate prefix (/odata/v4/assign-material-to-formula vs
        # /odata/v4/fertilizer-blend/...) — these differ, so they are joined normally.
        assert url == (
            "https://approuter.host"
            "/odata/v4/assign-material-to-formula"
            "/odata/v4/fertilizer-blend"
            "/FertilizerBlend(1856)"
            "/FertilizerBlendService.RefreshProcessOrder"
        )

    def test_executor_bound_action_collapsed_base(self):
        """Base whose path IS the prefix of the endpoint → prefix is collapsed."""
        tool = make_tool(
            binding=ToolBinding.BOUND,
            tool_type=ToolType.ACTION,
            http_endpoint="/odata/v4/fertilizer-blend/FertilizerBlend(<key>)/FertilizerBlendService.RefreshProcessOrder",
        )
        url = _exec._build_url(
            tool,
            {},
            "1856",
            "https://approuter.host/odata/v4/fertilizer-blend",
        )
        assert url == (
            "https://approuter.host"
            "/odata/v4/fertilizer-blend"
            "/FertilizerBlend(1856)"
            "/FertilizerBlendService.RefreshProcessOrder"
        )

    def test_executor_relative_base_raises_endpoint_resolution(self):
        """Relative service_base_url → EndpointResolutionError (never UNEXPECTED_ERROR)."""
        tool = make_tool(
            binding=ToolBinding.BOUND,
            tool_type=ToolType.ACTION,
            http_endpoint="/odata/v4/fertilizer-blend/FertilizerBlend(<key>)/FertilizerBlendService.RefreshProcessOrder",
        )
        with pytest.raises(EndpointResolutionError):
            _exec._build_url(
                tool,
                {},
                "1856",
                "/odata/v4/assign-material-to-formula",
            )
