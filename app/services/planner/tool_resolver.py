"""ToolResolver — match a registered tool to a TOOL_EXECUTION message and compute
the missing required parameters.

NO LLM. The ONLY Planner component that touches the DB, and only via the injected
AsyncSession + the existing `tool_catalog_service` read API (behind a
`ToolRepository` port so tests inject a fake — no DB, no network).

Matching is purely lexical: tool key/name/displayName overlap, action-verb ↔ tool
semantics, and binding the verb to the resolved entity. `missingParameters` is a
lexical presence check of each required parameter against the message and the
current Fiori record — it never extracts/guesses values with an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from app.models.tool_catalog import ToolDefinition
from app.services.planner import text_signals as ts

# Generic key-ish tokens that don't, on their own, prove a parameter was supplied.
_GENERIC_PARAM_TOKENS = {"id", "key", "code", "number", "no", "guid", "uuid", "ref"}
_MATCH_THRESHOLD = 0.5


class ToolRepository(Protocol):
    """Port over the Tool Registry read API."""

    async def list_tools(self, session, app_id: str) -> List[ToolDefinition]: ...


class ToolCatalogRepository:
    """Default ToolRepository delegating to `app.services.tool_catalog_service`."""

    async def list_tools(self, session, app_id: str) -> List[ToolDefinition]:
        from app.services import tool_catalog_service as svc
        return await svc.list_tools(session, app_id)


@dataclass
class ResolvedTool:
    tool_key: Optional[str] = None
    name: Optional[str] = None
    display_name: Optional[str] = None
    tool_type: Optional[str] = None
    score: float = 0.0
    matched_verbs: List[str] = field(default_factory=list)
    missing_parameters: List[str] = field(default_factory=list)


def _enum_value(v) -> Optional[str]:
    return getattr(v, "value", v) if v is not None else None


def _name_stems(text: Optional[str]) -> set:
    if not text:
        return set()
    parts = ts.ent_slug(text).replace("_", "-").split("-")
    return {ts.stem(p) for p in parts if p and p not in ts.NOISE}


class ToolResolver:
    """Stateless resolver. Construct once with a ToolRepository."""

    def __init__(self, repo: ToolRepository):
        self._repo = repo

    async def resolve(
        self,
        session,
        signals: ts.PlanSignals,
        app_id: Optional[str],
        resolved_entity: Optional[str] = None,
        fiori_context: Optional[Dict] = None,
    ) -> ResolvedTool:
        # No app or no DB session → cannot consult the Tool Registry. No-op.
        if not app_id or session is None:
            return ResolvedTool()

        try:
            tools = await self._repo.list_tools(session, app_id)
        except Exception:
            return ResolvedTool()
        if not tools:
            return ResolvedTool()

        verbs = sorted(ts.matched_mutating_verbs(signals.tokens))
        content_stems = {ts.stem(t) for t in signals.content_tokens}
        msg_tokens = set(signals.tokens)
        entity_low = resolved_entity.lower() if resolved_entity else None

        best: Optional[ToolDefinition] = None
        best_score = 0.0
        for tool in tools:
            score = self._score_tool(tool, signals, msg_tokens, content_stems, verbs, entity_low)
            if score > best_score:
                best_score = score
                best = tool

        if best is None or best_score < _MATCH_THRESHOLD:
            return ResolvedTool(matched_verbs=verbs)

        missing = self._missing_parameters(best, signals, fiori_context)
        return ResolvedTool(
            tool_key=best.tool_key,
            name=best.name,
            display_name=best.display_name,
            tool_type=_enum_value(best.tool_type),
            score=round(best_score, 4),
            matched_verbs=verbs,
            missing_parameters=missing,
        )

    # ── scoring ────────────────────────────────────────────────────────────────

    def _score_tool(
        self,
        tool: ToolDefinition,
        signals: ts.PlanSignals,
        msg_tokens: set,
        content_stems: set,
        verbs: List[str],
        entity_low: Optional[str],
    ) -> float:
        score = 0.0
        key_suffix = (tool.tool_key or "").split(".")[-1].lower()
        op_name = (tool.name or "").lower()

        # Exact operation-name / key-suffix mention in the message.
        if (key_suffix and key_suffix in msg_tokens) or (op_name and op_name in msg_tokens):
            score += ts.STRONG

        # name / display_name stem overlap with the message.
        tool_stems = _name_stems(tool.name) | _name_stems(tool.display_name) | _name_stems(key_suffix)
        if tool_stems:
            overlap = len(tool_stems & content_stems) / len(tool_stems)
            score += overlap * ts.MEDIUM

        # Action verb appearing inside the tool's own identifiers/description.
        haystack = " ".join(
            ts.normalize(x or "")
            for x in (tool.tool_key, tool.name, tool.display_name, tool.description)
        )
        if any(v in haystack for v in verbs):
            score += ts.STRONG

        # Binding the verb to the resolved entity.
        if entity_low:
            for attr in (tool.entity_name, tool.bound_entity, tool.name, tool.tool_key):
                if attr and entity_low in attr.lower():
                    score += ts.MEDIUM
                    break

        # An ACTION with a mutating verb present is a stronger TOOL_EXECUTION fit.
        if verbs and _enum_value(tool.tool_type) == "ACTION":
            score += ts.WEAK

        return score

    # ── missing parameters ──────────────────────────────────────────────────────

    def _missing_parameters(
        self,
        tool: ToolDefinition,
        signals: ts.PlanSignals,
        fiori_context: Optional[Dict],
    ) -> List[str]:
        required = list(tool.required_parameters or [])
        if not required:
            required = [p.name for p in (tool.parameters or []) if getattr(p, "required", False)]
        if not required:
            return []

        record = {}
        if fiori_context:
            ed = fiori_context.get("entity_data") or fiori_context.get("entityData")
            if isinstance(ed, dict):
                record = {str(k).lower(): v for k, v in ed.items()}

        content_stems = {ts.stem(t) for t in signals.content_tokens}
        has_literal = self._has_value_literal(signals)

        unresolved: List[str] = []
        for name in required:
            if self._param_present(name, content_stems, record):
                continue
            unresolved.append(name)

        # Conservative single-literal fill: if exactly one required param is
        # unresolved and the message carries a standalone value literal, assume
        # it fills that param (no multi-param guessing).
        if len(unresolved) == 1 and has_literal:
            return []
        return unresolved

    @staticmethod
    def _param_present(name: str, content_stems: set, record: Dict) -> bool:
        if name.lower() in record and record[name.lower()] not in (None, "", []):
            return True
        parts = ts.ent_slug(name).replace("_", "-").split("-")
        meaningful = [ts.stem(p) for p in parts if p and p not in _GENERIC_PARAM_TOKENS and p not in ts.NOISE]
        if meaningful and (set(meaningful) & content_stems):
            return True
        return False

    @staticmethod
    def _has_value_literal(signals: ts.PlanSignals) -> bool:
        import re
        if '"' in signals.raw or "'" in signals.raw:
            return True
        # a number or an alphanumeric code like B-100 / PO4711
        return re.search(r"\b\d+\b|\b[A-Za-z]+[-_]?\d+\b", signals.raw) is not None
