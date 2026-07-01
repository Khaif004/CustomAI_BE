"""MetadataRetriever — application metadata from the in-memory service registry.

Reuses `app.api.apps._service_tool_registry` (the same registry the Planner's
EntityResolver reads). Read-only, fully in-memory (no DB, no executor). Emits one
item per service, per entity (with its fields), and per association.

`entity_aliases` / `entity_associations` are only present on DB-loaded registry
entries (live-registered entries omit them), so they are read defensively with
`.get(...)`.
"""
from __future__ import annotations

import logging
from typing import ClassVar, List

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


class MetadataRetriever(Retriever):
    source: ClassVar[RetrievalSource] = RetrievalSource.METADATA
    section: ClassVar[Section] = Section.METADATA

    async def retrieve(self, request: RetrievalRequest) -> RetrieverResult:
        app_id = request.app_id
        if not app_id:
            return self._empty()
        try:
            from app.api.apps import _service_tool_registry  # lazy: avoid import cost/cycles
            entries = list(_service_tool_registry.get(app_id, []) or [])
        except Exception as e:  # registry unavailable — degrade, don't crash
            return self._empty(error=str(e))

        items: List[RetrievalItem] = []
        seen: set = set()

        def _add(item: RetrievalItem) -> None:
            if item.ref in seen:
                return
            seen.add(item.ref)
            items.append(item)

        for entry in entries:
            service_url = entry.get("service_url", "")
            entity_fields = entry.get("entity_fields", {}) or {}
            entities = entry.get("entities", []) or []
            associations = entry.get("entity_associations", []) or []
            aliases = entry.get("entity_aliases", {}) or {}

            _add(RetrievalItem(
                source=self.source, tier=TIER_EXACT,
                ref=f"service:{service_url}",
                title=entry.get("app_name") or app_id,
                content=service_url or None,
                data={
                    "kind": "service",
                    "service_url": service_url,
                    "app_base_url": entry.get("app_base_url", ""),
                    "entity_count": len(entities),
                },
            ))

            for ename in entities:
                short = str(ename).split(".")[-1]
                _add(RetrievalItem(
                    source=self.source, tier=TIER_EXACT,
                    ref=f"entity:{short}",
                    title=short,
                    data={
                        "kind": "entity",
                        "entity": short,
                        "fields": entity_fields.get(ename) or entity_fields.get(short) or [],
                        "service_url": service_url,
                    },
                ))

            for assoc in associations:
                if not isinstance(assoc, dict):
                    continue
                _add(RetrievalItem(
                    source=self.source, tier=TIER_EXACT,
                    ref=f"assoc:{assoc.get('source')}:{assoc.get('fk_field')}:{assoc.get('target')}",
                    title=f"{assoc.get('source')} → {assoc.get('target')}",
                    data={"kind": "association", **assoc},
                ))

            for alias, canonical in aliases.items():
                _add(RetrievalItem(
                    source=self.source, tier=TIER_EXACT,
                    ref=f"alias:{alias}",
                    title=str(alias),
                    data={"kind": "alias", "alias": alias, "entity": canonical},
                ))

        return self._result(items)
