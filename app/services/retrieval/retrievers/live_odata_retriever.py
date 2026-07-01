"""LiveODataRetriever — deterministic, non-LLM live OData fetch.

Runs only when the plan requires live data (the orchestrator schedules it when
`LiveOData` is among the plan's sources; it also re-checks
`plan.requires_live_data`). It performs a single `$top=N&$count=true` GET for the
plan's already-resolved entity, mirroring `chat_agent._query_entity` — NO LLM is
involved in selecting the entity, building the URL, or fetching.

No shared async OData helper exists to import, so the HTTP GET is a thin,
self-contained `aiohttp` call. It is INJECTABLE (`fetcher=`) so unit tests run
with zero network. Everything is best-effort: any failure → empty `LiveData`
section, never an exception.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable, ClassVar, Dict, List, Optional

from app.models.planner import RetrievalSource
from app.services.retrieval.base import Retriever
from app.services.retrieval.models import (
    TIER_EXACT,
    RetrievalItem,
    RetrievalRequest,
    RetrieverResult,
    Section,
)

logger = logging.getLogger(__name__)

# fetcher(base, entity_set, headers, top) -> {"set","rows","count"} | None
Fetcher = Callable[[str, str, Dict[str, str], int], Awaitable[Optional[Dict[str, Any]]]]


async def _aiohttp_fetch(base: str, entity_set: str, headers: Dict[str, str], top: int):
    import aiohttp
    url = f"{base}/{entity_set}"
    params = {"$top": str(min(top, 20)), "$count": "true"}
    timeout = aiohttp.ClientTimeout(total=6)
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    rows = data.get("value", []) if isinstance(data, dict) else []
    total = data.get("@odata.count") if isinstance(data, dict) else None
    return {"set": entity_set, "rows": rows[: min(top, 20)],
            "count": int(total) if total is not None else None}


class LiveODataRetriever(Retriever):
    source: ClassVar[RetrievalSource] = RetrievalSource.LIVE_ODATA
    section: ClassVar[Section] = Section.LIVE_DATA

    def __init__(self, fetcher: Optional[Fetcher] = None):
        self._fetch = fetcher or _aiohttp_fetch

    async def retrieve(self, request: RetrievalRequest) -> RetrieverResult:
        # Trust the plan — never re-decide. (Orchestrator already gates on the source.)
        if not request.plan.requires_live_data:
            return self._empty()

        entity = request.plan.entity or self._entity_from_schema_hint(request.fiori_context)
        if not entity:
            return self._empty()

        base = self._resolve_base(request)
        if not base:
            return self._empty()

        headers = {"Accept": "application/json", "OData-MaxVersion": "4.0"}
        token = self._token(request)
        if token:
            raw = token.replace("Bearer ", "").replace("bearer ", "")
            headers["Authorization"] = f"Bearer {raw}"

        try:
            fetched = None
            for candidate in self._set_candidates(entity):
                fetched = await self._fetch(base, candidate, headers, request.top)
                if fetched is not None:
                    break
        except Exception as e:
            return self._empty(error=str(e))

        if not fetched:
            return self._empty()

        set_name = fetched.get("set", entity)
        count = fetched.get("count")
        rows = fetched.get("rows", []) or []
        content = (
            f"{set_name}: {count} record(s)" if count is not None
            else f"{set_name}: {len(rows)} row(s) fetched"
        )
        item = RetrievalItem(
            source=self.source, tier=TIER_EXACT,
            ref=f"live:{set_name}",
            title=set_name,
            content=content,
            data={"entity_set": set_name, "count": count, "rows": rows},
        )
        return self._result([item])

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _entity_from_schema_hint(fiori_context: Optional[Dict]) -> Optional[str]:
        if not fiori_context:
            return None
        extra = fiori_context.get("extra") if isinstance(fiori_context.get("extra"), dict) else {}
        hint = (extra or {}).get("schema_hint")
        if not hint:
            return None
        for line in str(hint).splitlines():
            m = re.match(r"^Entity:\s*(\w+)", line.strip())
            if m:
                return m.group(1)
            m2 = re.match(r"^Entities:\s*(\w+)", line.strip())
            if m2:
                return m2.group(1)
        return None

    @staticmethod
    def _token(request: RetrievalRequest) -> Optional[str]:
        fc = request.fiori_context or {}
        return fc.get("odata_token") or request.odata_token

    def _resolve_base(self, request: RetrievalRequest) -> Optional[str]:
        """Resolve an absolute OData service base URL (relative → absolute)."""
        fc = request.fiori_context or {}
        service_url = fc.get("service_url") or fc.get("serviceUrl")

        registry_base = ""
        if request.app_id:
            try:
                from app.api.apps import get_service_tool
                tool = get_service_tool(request.app_id)
                if tool:
                    registry_base = tool.get("app_base_url", "") or ""
                    if not service_url:
                        service_url = tool.get("service_url")
            except Exception:
                pass

        if not service_url:
            return None
        if service_url.startswith("http"):
            return service_url.rstrip("/")

        # relative → resolve a base
        base = registry_base
        if not base:
            try:
                from app.config import get_settings
                base = (get_settings().cap_app_base_url or "").rstrip("/")
            except Exception:
                base = ""
        if not base:
            extra = fc.get("extra") if isinstance(fc.get("extra"), dict) else {}
            page_url = (extra or {}).get("page_url", "")
            if page_url:
                try:
                    from urllib.parse import urlparse
                    p = urlparse(page_url)
                    base = f"{p.scheme}://{p.netloc}"
                except Exception:
                    base = ""
        if not base:
            return None
        return f"{base.rstrip('/')}{service_url}".rstrip("/")

    @staticmethod
    def _set_candidates(entity: str) -> List[str]:
        """Entity TYPE → OData SET-name candidates (mirrors _resolve_set_name)."""
        cands = [entity]
        if not entity.endswith("s"):
            cands.append(entity + "s")
        else:
            cands.append(entity[:-1])
        # de-dupe, preserve order
        seen, out = set(), []
        for c in cands:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out
