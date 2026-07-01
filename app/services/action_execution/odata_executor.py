"""Low-level OData executor: builds URL, executes HTTP, normalises response.

Security invariant
------------------
The only URL components accepted come from the Tool Registry (http_endpoint)
and the Application Registry (app_base_url stored in the applications table).
User-supplied parameters are placed into POST bodies or query strings only —
never concatenated into the URL path beyond the entity-key substitution that
uses a pre-defined template from the registry.
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

import aiohttp

from app.models.tool_catalog import ToolBinding, ToolDefinition, ToolType
from app.services.action_execution.exceptions import (
    ConfigurationError,
    EndpointResolutionError,
    ODataExecutionError,
)

logger = logging.getLogger("action_execution.odata")

# ── tunables ──────────────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT_SEC: int = 120
_MAX_RETRIES: int = 2
_RETRY_STATUSES = frozenset({502, 503, 504})
_RETRY_BACKOFF: List[float] = [0.5, 1.0]

# Supported placeholder patterns in http_endpoint templates for bound ops.
# Matches: <key>, {key}, {KEY}, <KEY>
_KEY_PLACEHOLDER_RE = re.compile(r"[<{]key[}>]", re.IGNORECASE)


@dataclass
class ODataRawResponse:
    """Normalised result from a single OData HTTP call."""

    http_status: int
    result: Optional[Any]
    messages: List[str]
    raw_url: str
    duration_ms: float


# ── URL construction ──────────────────────────────────────────────────────────

def build_absolute_url(app_base_url: str, http_endpoint: str) -> str:
    """Join the app base URL with an OData endpoint path into an absolute URL.

    Rules
    -----
    * ``app_base_url`` must contain a scheme and hostname.  If it does not,
      ``EndpointResolutionError`` is raised with a diagnostic message — the raw
      ``aiohttp.InvalidURL`` exception is never allowed to leak.
    * Trailing/leading slashes are normalised so joining never produces ``//``.
    * Duplicate OData path prefix is collapsed automatically: if the *path
      component* of ``app_base_url`` is already a leading prefix of
      ``http_endpoint``, the endpoint is appended directly to the scheme+host
      origin — preventing paths like ``/odata/v4/svc/odata/v4/svc/Entity``.

    Examples
    --------
    >>> build_absolute_url("https://host", "/odata/v4/svc/Foo")
    'https://host/odata/v4/svc/Foo'

    >>> build_absolute_url("https://host/", "odata/v4/svc/Foo")
    'https://host/odata/v4/svc/Foo'

    >>> build_absolute_url("https://host/odata/v4/svc", "/odata/v4/svc/Foo")
    'https://host/odata/v4/svc/Foo'          # duplicate prefix collapsed

    >>> build_absolute_url("https://host/app", "/odata/v4/svc/Foo")
    'https://host/app/odata/v4/svc/Foo'      # different paths, joined normally
    """
    if not app_base_url:
        raise ConfigurationError(
            "No app base URL configured. "
            "Ensure the CAP application has registered app_base_url via "
            "/api/apps/register-tools or /api/apps/register-service-tool."
        )

    base = app_base_url.rstrip("/")
    # Guarantee the endpoint always has a leading slash before joining.
    endpoint = "/" + http_endpoint.lstrip("/")

    parsed_base = urlparse(base)
    # Path component of the base URL — never has a trailing slash after rstrip.
    base_path = parsed_base.path.rstrip("/")

    # Duplicate-prefix detection: if the base path is a non-empty leading prefix
    # of the endpoint, the service registry has already embedded the OData
    # service path — collapse it so we don't repeat it.
    if base_path and (
        endpoint.lower().startswith(base_path.lower() + "/")
        or endpoint.lower() == base_path.lower()
    ):
        origin = f"{parsed_base.scheme}://{parsed_base.netloc}"
        full_url = origin + endpoint
    else:
        full_url = base + endpoint

    # Validate: must be an absolute URL with both scheme and hostname.
    parsed = urlparse(full_url)
    if not parsed.scheme or not parsed.hostname:
        raise EndpointResolutionError(
            "Cannot build an absolute URL for tool execution.\n"
            f"  App base URL     : {app_base_url!r}\n"
            f"  Tool endpoint    : {http_endpoint!r}\n"
            f"  Resolved URL     : {full_url!r}\n"
            "The app base URL must be an absolute URL with a scheme and hostname "
            "(e.g. 'https://tenant.cfapps.eu10.hana.ondemand.com'). "
            "Update the app_base_url in the tool registration for this application."
        )

    return full_url


def _extract_missing_key_field(error_message: str) -> Optional[str]:
    """Return the key field name from a CAP 'Key "X" is missing' error, or None."""
    m = re.search(r'Key "(\w+)" is missing', error_message)
    return m.group(1) if m else None


# Well-known CAP/OData composite-key fields and their production defaults.
# IsActiveEntity is injected by CAP's draft-enabled entities; true = active record.
_CAP_KEY_DEFAULTS: Dict[str, str] = {
    "IsActiveEntity": "true",
}


def _add_missing_key(url: str, entity_key: str, key_field: str) -> str:
    """Fix a 'Key X is missing' URL in-place.

    Two cases:
    1. Positional key ``(value)`` → convert to named ``(key_field=value)``.
    2. Already-named key ``(f1=v1,...)`` → append ``,key_field=default``.
    """
    escaped = re.escape(entity_key)

    # Case 1: bare positional value — the very first key segment
    positional = re.compile(rf'\({escaped}\)(?=/|$)')
    if positional.search(url):
        return positional.sub(f'({key_field}={entity_key})', url, count=1)

    # Case 2: already has named keys — append the missing field using its default
    default_val = _CAP_KEY_DEFAULTS.get(key_field)
    if default_val is None:
        return url  # unknown field, no safe default — caller will re-raise

    def _append(m: re.Match) -> str:
        return f"({m.group(1)},{key_field}={default_val})"

    return re.sub(r'\(([^)]*=[^)]*)\)(?=/|$)', _append, url, count=1)


class ODataExecutor:
    """Builds and executes OData v4 requests for registered actions/functions.

    Inject a custom timeout_sec for tests or high-latency CAP deployments.
    """

    def __init__(self, timeout_sec: int = _DEFAULT_TIMEOUT_SEC) -> None:
        self._timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async def execute(
        self,
        tool: ToolDefinition,
        parameters: Dict[str, Any],
        entity_key: Optional[str],
        app_base_url: str,
        odata_token: Optional[str] = None,
    ) -> ODataRawResponse:
        """Build URL, execute the OData request, return normalised response."""
        url = self._build_url(tool, parameters, entity_key, app_base_url)
        headers = self._build_headers(odata_token)
        method = (tool.http_method or "POST").upper()

        logger.info("[odata_executor] App Base URL     : %s", app_base_url)
        logger.info("[odata_executor] Tool Endpoint    : %s", tool.http_endpoint)
        logger.info("[odata_executor] Resolved URL     : %s", url)
        logger.info("[odata_executor] Method           : %s", method)

        # Actions → POST with JSON body; Functions → GET (params already in URL)
        body: Optional[Dict[str, Any]] = None
        if method == "POST":
            body = {k: v for k, v in parameters.items() if v is not None}

        # Retry loop: CAP draft-enabled entities use composite keys.
        # Each iteration adds one missing key field and re-fires the request.
        # Cap at 4 iterations so we never loop forever.
        current_url = url
        for _fix_attempt in range(4):
            try:
                return await self._execute_with_retry(current_url, method, headers, body)
            except ODataExecutionError as exc:
                if exc.status_code != 400 or tool.binding != ToolBinding.BOUND or not entity_key:
                    raise
                key_field = _extract_missing_key_field(str(exc))
                if not key_field:
                    raise
                new_url = _add_missing_key(current_url, entity_key, key_field)
                if new_url == current_url:
                    raise  # no progress — key has no known default, give up
                logger.info(
                    "[odata_executor] Auto-adding key '%s': %s", key_field, new_url
                )
                current_url = new_url
        raise ODataExecutionError(400, "Could not resolve all entity key fields after auto-fix.")

    # ── URL building ──────────────────────────────────────────────────────────

    def _build_url(
        self,
        tool: ToolDefinition,
        parameters: Dict[str, Any],
        entity_key: Optional[str],
        app_base_url: str,
    ) -> str:
        if not app_base_url:
            raise ConfigurationError(
                "No app base URL configured for this app. "
                "Ensure the CAP application has registered app_base_url via "
                "/api/apps/register-tools or /api/apps/register-service-tool."
            )
        if not tool.http_endpoint:
            raise EndpointResolutionError(
                f"Tool '{tool.tool_key}' has no http_endpoint defined in the registry."
            )

        endpoint = tool.http_endpoint

        # Bound operations: substitute the entity-key placeholder
        if tool.binding == ToolBinding.BOUND:
            if not entity_key:
                raise EndpointResolutionError(
                    f"Tool '{tool.tool_key}' is a bound operation and requires "
                    f"an entity_key in the request."
                )
            endpoint = _KEY_PLACEHOLDER_RE.sub(entity_key, endpoint)

        # Build and validate the absolute URL.
        # Raises EndpointResolutionError when app_base_url is relative or malformed.
        full_url = build_absolute_url(app_base_url, endpoint)

        # OData Functions: append parameters as query string or inline params
        if tool.tool_type == ToolType.FUNCTION:
            full_url = self._append_function_params(full_url, parameters)

        return full_url

    def _append_function_params(
        self, url: str, parameters: Dict[str, Any]
    ) -> str:
        """Append function parameters as a query string.

        If the endpoint template already contains inline params like
        FunctionName(param=value), skip appending (CAP supports both styles).
        """
        if not parameters:
            return url
        # If the URL already ends with (...contents...), params are embedded
        if re.search(r"\([^)]+\)\s*$", url):
            return url
        filtered = {k: v for k, v in parameters.items() if v is not None}
        if not filtered:
            return url
        qs = urlencode(filtered)
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}{qs}"

    # ── HTTP ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_headers(odata_token: Optional[str]) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "OData-MaxVersion": "4.0",
        }
        if odata_token:
            raw = odata_token.replace("Bearer ", "").replace("bearer ", "").strip()
            headers["Authorization"] = f"Bearer {raw}"
        return headers

    async def _execute_with_retry(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]],
    ) -> ODataRawResponse:
        last_status: int = 0

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                await asyncio.sleep(_RETRY_BACKOFF[attempt - 1])

            t0 = time.monotonic()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method,
                        url,
                        json=body if method == "POST" else None,
                        headers=headers,
                        timeout=self._timeout,
                    ) as resp:
                        duration_ms = (time.monotonic() - t0) * 1000
                        last_status = resp.status

                        # Transient gateway errors → retry
                        if resp.status in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                            continue

                        return await self._parse_response(resp, url, duration_ms)

            except (asyncio.TimeoutError, TimeoutError):
                # Never retry POST timeouts — the server may already be executing the
                # operation and a retry would cause duplicate side effects (e.g. two
                # process orders created). GET/HEAD are safe to retry on timeout.
                if attempt < _MAX_RETRIES and method != "POST":
                    continue
                raise asyncio.TimeoutError(
                    f"OData request to '{url}' timed out after {self._timeout.total}s."
                )

            except aiohttp.InvalidURL as exc:
                # aiohttp raises this when given a relative or otherwise malformed URL.
                # build_absolute_url should prevent this, but defend in depth.
                raise EndpointResolutionError(
                    f"Malformed URL for OData request: '{url}'. "
                    f"Detail: {exc}. "
                    f"The service base URL in the registry must be absolute "
                    f"(e.g. 'https://tenant.cfapps.eu10.hana.ondemand.com')."
                )

            except aiohttp.ClientConnectorError as exc:
                if attempt < _MAX_RETRIES:
                    continue
                raise ODataExecutionError(0, f"Cannot reach OData service: {exc}")

        raise ODataExecutionError(last_status, "Request failed after retries.")

    # ── response normalisation ────────────────────────────────────────────────

    @staticmethod
    async def _parse_response(
        resp: aiohttp.ClientResponse,
        url: str,
        duration_ms: float,
    ) -> ODataRawResponse:
        messages: List[str] = []

        # 204 No Content — action succeeded, nothing to deserialise
        if resp.status == 204:
            return ODataRawResponse(
                http_status=204,
                result=None,
                messages=["Action executed successfully."],
                raw_url=url,
                duration_ms=duration_ms,
            )

        # 4xx / 5xx — extract OData error detail and raise
        if resp.status >= 400:
            text = await resp.text()
            detail = text[:500]
            try:
                err_body = _json.loads(text)
                detail = (
                    err_body.get("error", {}).get("message")
                    or err_body.get("message")
                    or detail
                )
            except Exception:
                pass
            raise ODataExecutionError(resp.status, detail)

        # 2xx — parse response body
        content_type = resp.headers.get("Content-Type", "")
        result: Optional[Any] = None

        if "application/json" in content_type:
            try:
                data = await resp.json(content_type=None)
            except Exception:
                data = await resp.text()

            if isinstance(data, dict):
                # Unwrap OData value array if present; drop metadata keys
                result = data.get("value", data)
                for meta_key in ("@odata.context", "@odata.metadataEtag"):
                    if isinstance(result, dict):
                        result.pop(meta_key, None)
                # Surface SAP Business Application Studio messages
                for msg in data.get("@sap.messages", []):
                    if isinstance(msg, dict):
                        messages.append(msg.get("message", str(msg)))
            else:
                result = data
        else:
            result = await resp.text()

        return ODataRawResponse(
            http_status=resp.status,
            result=result,
            messages=messages,
            raw_url=url,
            duration_ms=duration_ms,
        )
