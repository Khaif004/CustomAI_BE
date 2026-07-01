"""ContextBuilder — transform a RetrievalContext into a structured LLMContext.

Pipeline (NO retrieval, NO planning, NO prompts, NO LLM):

    RetrievalContext
      → gather   : RetrievalItem → ContextItem (provenance + token estimate),
                   routed to its target section by source
      → dedup    : exact-content + subsumption ("metadata defines field" wins
                   over "semantic doc describing it"); higher-priority source kept
      → rank     : within each section by (exact, confidence, score, freshness)
      → system   : derive structured (data-only) capability directives
      → budget   : enforce per-section caps then the global max, trimming the
                   lowest-priority sections / lowest-ranked items first
      → stats    : token estimate, retrievers used, discarded, duplicates, ratio
    → LLMContext

It accepts ONLY a RetrievalContext (+ optional settings override). It is stateless
and model-agnostic: the injected `TokenEstimator` is the only thing that could be
model-specific, and the default is provider-neutral — so this layer supports SAP
AI Core / OpenAI / Claude / Gemini / local LLMs without modification. The
model-specific concern (prompt formatting) belongs to a future Prompt Builder.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.models.llm_context import (
    CONTEXT_BUILDER_SOURCE,
    ContextBuilderSettings,
    ContextItem,
    ContextSection,
    ContextStatistics,
    LLMContext,
)
from app.models.planner import RetrievalSource
from app.services.context_builder.tokenizer import DEFAULT_TOKEN_ESTIMATOR, TokenEstimator
from app.services.retrieval.models import RetrievalContext, RetrievalItem

logger = logging.getLogger(__name__)

# ── Source → priority / section / retriever (priority 1 = highest, kept longest) ──
# Mirrors the spec priority list: LiveOData > Metadata > Tool > Documentation >
# Conversation > Semantic(pgvector) > Keyword(FTS). UI context is treated as
# very-high-value current context; CodeSummaries is forward-declared.
_SOURCE_PRIORITY: Dict[RetrievalSource, int] = {
    RetrievalSource.LIVE_ODATA: 1,
    RetrievalSource.UI_CONTEXT: 2,
    RetrievalSource.METADATA: 3,
    RetrievalSource.TOOL_REGISTRY: 4,
    RetrievalSource.DOCUMENTATION: 5,
    RetrievalSource.CONVERSATION_MEMORY: 6,
    RetrievalSource.PGVECTOR: 7,
    RetrievalSource.FULL_TEXT_SEARCH: 8,
    RetrievalSource.CODE_SUMMARIES: 9,
}

_SOURCE_TO_SECTION: Dict[RetrievalSource, ContextSection] = {
    RetrievalSource.LIVE_ODATA: ContextSection.LIVE_BUSINESS_DATA,
    RetrievalSource.METADATA: ContextSection.APPLICATION_METADATA,
    RetrievalSource.TOOL_REGISTRY: ContextSection.TOOL_METADATA,
    RetrievalSource.DOCUMENTATION: ContextSection.DOCUMENTATION,
    RetrievalSource.CONVERSATION_MEMORY: ContextSection.CONVERSATION_CONTEXT,
    RetrievalSource.UI_CONTEXT: ContextSection.CURRENT_UI_CONTEXT,
    RetrievalSource.PGVECTOR: ContextSection.SEMANTIC_KNOWLEDGE,
    RetrievalSource.FULL_TEXT_SEARCH: ContextSection.SEMANTIC_KNOWLEDGE,
    RetrievalSource.CODE_SUMMARIES: ContextSection.SEMANTIC_KNOWLEDGE,
}

_SOURCE_TO_RETRIEVER: Dict[RetrievalSource, str] = {
    RetrievalSource.LIVE_ODATA: "LiveODataRetriever",
    RetrievalSource.METADATA: "MetadataRetriever",
    RetrievalSource.TOOL_REGISTRY: "ToolRetriever",
    RetrievalSource.DOCUMENTATION: "DocumentationRetriever",
    RetrievalSource.CONVERSATION_MEMORY: "MemoryRetriever",
    RetrievalSource.UI_CONTEXT: "UIContextRetriever",
    RetrievalSource.PGVECTOR: "VectorRetriever",
    RetrievalSource.FULL_TEXT_SEARCH: "KeywordRetriever",
    RetrievalSource.CODE_SUMMARIES: "CodeRetriever",
}

# Each section's priority = the highest-priority (lowest number) source that maps
# to it. Derived from _SOURCE_PRIORITY so the two can never drift apart.
_SECTION_PRIORITY: Dict[ContextSection, int] = {}
for _src, _sec in _SOURCE_TO_SECTION.items():
    _p = _SOURCE_PRIORITY[_src]
    if _sec not in _SECTION_PRIORITY or _p < _SECTION_PRIORITY[_sec]:
        _SECTION_PRIORITY[_sec] = _p

# Global trim order: lowest-priority sections trimmed FIRST (highest priority
# NUMBER first), derived from _SECTION_PRIORITY so it stays in sync — the previous
# hand-written list had a priority inversion (LiveBusinessData before UI context).
# SystemInstructions is never trimmed for the global budget.
_TRIM_ORDER: List[ContextSection] = sorted(
    (s for s in ContextSection if s != ContextSection.SYSTEM_INSTRUCTIONS),
    key=lambda s: _SECTION_PRIORITY.get(s, 0),
    reverse=True,
)

_DOC_SECTIONS = {ContextSection.SEMANTIC_KNOWLEDGE, ContextSection.DOCUMENTATION}
_TIMESTAMP_KEYS = (
    "timestamp", "registered_at", "registeredAt", "modified_at", "modifiedAt",
    "updated_at", "updatedAt", "created_at", "createdAt", "lastModified",
)


@dataclass
class _Entry:
    item: ContextItem
    section: ContextSection
    priority: int


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _freshness(ts: Optional[str]) -> float:
    """Best-effort freshness score (epoch seconds; newer = larger). Naive ISO
    timestamps are treated as UTC so they share one absolute scale with tz-aware
    ones (otherwise local-offset skew could flip ordering of otherwise-tied items).
    Freshness is only the lowest-priority tie-break, so unparseable → 0.0."""
    if not ts:
        return 0.0
    try:
        from datetime import datetime, timezone
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except Exception:
        return 0.0


class ContextBuilder:
    def __init__(
        self,
        settings: Optional[ContextBuilderSettings] = None,
        tokenizer: Optional[TokenEstimator] = None,
    ):
        self._settings = settings or ContextBuilderSettings()
        self._tok = tokenizer or DEFAULT_TOKEN_ESTIMATOR

    def build(
        self,
        context: RetrievalContext,
        settings: Optional[ContextBuilderSettings] = None,
    ) -> LLMContext:
        s = settings or self._settings

        entries = self._gather(context)
        original_tokens = sum(e.item.token_estimate for e in entries)

        survivors, duplicate_count = self._dedup(entries)

        sections: Dict[ContextSection, List[ContextItem]] = {sec: [] for sec in ContextSection}
        for e in self._rank(survivors):
            sections[e.section].append(e.item)

        if s.include_system_directives:
            sections[ContextSection.SYSTEM_INSTRUCTIONS] = self._system_directives(context)

        sections, discarded = self._apply_budget(sections, s)

        kept = [it for items in sections.values() for it in items]
        kept_tokens = sum(it.token_estimate for it in kept)
        kept_retrieval_tokens = sum(
            it.token_estimate for it in kept if it.source != CONTEXT_BUILDER_SOURCE
        )
        retrievers_used = sorted(
            {it.retriever for it in kept if it.retriever and it.source != CONTEXT_BUILDER_SOURCE}
        )
        stats = ContextStatistics(
            token_estimate=kept_tokens,
            retrievers_used=retrievers_used,
            documents_discarded=discarded,
            duplicate_count=duplicate_count,
            compression_ratio=(
                round(kept_retrieval_tokens / original_tokens, 4) if original_tokens else 1.0
            ),
        )

        return LLMContext(
            application=context.application,
            system_instructions=sections[ContextSection.SYSTEM_INSTRUCTIONS],
            application_metadata=sections[ContextSection.APPLICATION_METADATA],
            live_business_data=sections[ContextSection.LIVE_BUSINESS_DATA],
            tool_metadata=sections[ContextSection.TOOL_METADATA],
            documentation=sections[ContextSection.DOCUMENTATION],
            conversation_context=sections[ContextSection.CONVERSATION_CONTEXT],
            current_ui_context=sections[ContextSection.CURRENT_UI_CONTEXT],
            semantic_knowledge=sections[ContextSection.SEMANTIC_KNOWLEDGE],
            statistics=stats,
        )

    # ── gather ────────────────────────────────────────────────────────────────

    def _gather(self, context: RetrievalContext) -> List[_Entry]:
        all_items: List[RetrievalItem] = (
            list(context.live_data)
            + list(context.metadata)
            + list(context.tools)
            + list(context.documentation)
            + list(context.conversation_memory)
            + list(context.semantic_documents)
            + list(context.keyword_matches)
        )
        entries: List[_Entry] = []
        for ri in all_items:
            src = ri.source
            section = _SOURCE_TO_SECTION.get(src, ContextSection.SEMANTIC_KNOWLEDGE)
            ci = self._to_context_item(ri)
            entries.append(_Entry(item=ci, section=section, priority=_SOURCE_PRIORITY.get(src, 99)))
        return entries

    def _to_context_item(self, ri: RetrievalItem) -> ContextItem:
        data = ri.data if isinstance(ri.data, dict) else {}
        # Planner confidence is not part of RetrievalContext today; use it only if a
        # future retrieval layer embeds it, else fall back to the retriever score.
        planner_conf = data.get("planner_confidence")
        confidence = planner_conf if planner_conf is not None else ri.score
        ci = ContextItem(
            source=ri.source.value,
            retriever=_SOURCE_TO_RETRIEVER.get(ri.source),
            confidence=confidence,
            score=ri.score,
            timestamp=self._timestamp(data),
            exact=(ri.tier != "semantic"),
            ref=ri.ref,
            title=ri.title,
            content=ri.content,
            data=data,
        )
        ci.token_estimate = self._tok.estimate(self._text_of(ci))
        return ci

    @staticmethod
    def _timestamp(data: dict) -> Optional[str]:
        for k in _TIMESTAMP_KEYS:
            v = data.get(k)
            if v:
                return str(v)
        return None

    @staticmethod
    def _text_of(ci: ContextItem) -> str:
        parts = [ci.title or "", ci.content or ""]
        if ci.data:
            try:
                parts.append(json.dumps(ci.data, default=str, separators=(",", ":")))
            except Exception:
                parts.append(str(ci.data))
        return " ".join(p for p in parts if p)

    # ── dedup ─────────────────────────────────────────────────────────────────

    def _dedup(self, entries: List[_Entry]) -> Tuple[List[_Entry], int]:
        duplicates = 0

        # Pass A — exact duplicates: same content (docs) or same ref (structured).
        groups: Dict[str, _Entry] = {}
        order: List[str] = []
        for i, e in enumerate(entries):
            key = self._dedup_key(e.item, e.section)
            if key is None:                       # fully-empty item → never dedup
                key = f"u:{i}"
            kept = groups.get(key)
            if kept is None:
                groups[key] = e
                order.append(key)
                continue
            duplicates += 1
            # Keep the better item: prefer EXACT over semantic, then higher-priority
            # source, then higher score (so "prefer exact business data" holds even
            # when the duplicate text came from a semantic retriever).
            if self._dedup_rank(e) < self._dedup_rank(kept):
                groups[key] = e
        survivors = [groups[k] for k in order]

        # Pass B — subsumption: drop semantic/keyword docs already covered by an
        # exact metadata/tool/live definition (the spec's metadata-field example).
        covered = self._covered_terms(survivors)
        # Single-word covered terms (len >= 4) for whole-word TITLE containment — a
        # real semantic doc title is a sentence/heading ("The Status field …"), not
        # set-equal to the field name. We match the title (not full content) and only
        # longer tokens to limit false positives.
        covered_words = {c for c in covered if len(c) >= 4 and re.fullmatch(r"[a-z0-9]+", c)}
        result: List[_Entry] = []
        for e in survivors:
            if e.section == ContextSection.SEMANTIC_KNOWLEDGE:
                t = _norm(e.item.title) if e.item.title else None
                r = _norm(e.item.ref) if e.item.ref else None
                if (t and t in covered) or (r and r in covered):
                    duplicates += 1
                    continue
                if t and covered_words & set(re.findall(r"[a-z0-9]+", t)):
                    duplicates += 1
                    continue
            result.append(e)
        return result, duplicates

    @staticmethod
    def _dedup_rank(e: "_Entry") -> tuple:
        return (0 if e.item.exact else 1, e.priority, -(e.item.score or 0.0))

    @staticmethod
    def _dedup_key(ci: ContextItem, section: ContextSection) -> Optional[str]:
        # Identical document TEXT is deduped globally across sources (desirable —
        # e.g. a doc returned by both vector and keyword search). Structured items
        # are keyed by ref/title which RetrievalItem documents as a PER-SECTION key,
        # so those are namespaced by section to avoid cross-section collisions
        # (e.g. a bare tool_key colliding with another section's ref).
        if ci.content:
            return "c:" + hashlib.sha256(_norm(ci.content).encode("utf-8")).hexdigest()
        if ci.ref:
            return f"r:{section.value}:" + _norm(ci.ref)
        if not (ci.title or ci.data):
            return None                            # nothing to key on → unique
        blob = _norm(ci.title) + "|" + _norm(json.dumps(ci.data, default=str, sort_keys=True))
        return f"t:{section.value}:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _covered_terms(entries: List[_Entry]) -> set:
        covered: set = set()
        for e in entries:
            d = e.item.data or {}
            if e.section == ContextSection.APPLICATION_METADATA:
                ent = d.get("entity")
                if ent:
                    covered.add(_norm(ent))
                for f in (d.get("fields") or []):
                    fname = f.get("name") if isinstance(f, dict) else f  # tolerate dict fields
                    if not fname:
                        continue
                    covered.add(_norm(fname))
                    if ent:
                        covered.add(_norm(f"{ent}.{fname}"))
                if d.get("kind") == "association":
                    for kk in ("source", "target"):
                        if d.get(kk):
                            covered.add(_norm(d[kk]))
                if e.item.title:
                    covered.add(_norm(e.item.title))
            elif e.section == ContextSection.TOOL_METADATA:
                for val in (e.item.title, e.item.ref, d.get("name")):
                    if val:
                        covered.add(_norm(val))
            elif e.section == ContextSection.LIVE_BUSINESS_DATA:
                es = d.get("entity_set")
                if es:
                    covered.add(_norm(es))
        return covered

    # ── rank ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _rank(entries: List[_Entry]) -> List[_Entry]:
        # Within a section: exact first, then planner confidence, then retriever
        # score, then freshness (newest first). Cross-section grouping happens after.
        def key(e: _Entry):
            it = e.item
            return (
                0 if it.exact else 1,
                -(it.confidence if it.confidence is not None else (it.score or 0.0)),
                -(it.score or 0.0),
                -_freshness(it.timestamp),
            )
        # Stable sort keeps original relative order for full ties.
        return sorted(entries, key=key)

    # ── system directives (structured, data-only — NOT prompt text) ─────────────

    def _system_directives(self, context: RetrievalContext) -> List[ContextItem]:
        items: List[ContextItem] = []

        def add(kind: str, **fields):
            data = {"kind": kind, **fields}
            ci = ContextItem(
                source=CONTEXT_BUILDER_SOURCE,
                retriever=CONTEXT_BUILDER_SOURCE,
                exact=True,
                ref=f"directive:{kind}",
                title=kind,
                data=data,
            )
            ci.token_estimate = self._tok.estimate(json.dumps(data, default=str))
            items.append(ci)

        add(
            "grounding_policy",
            prefer_exact_business_data=True,
            # Derived from _SOURCE_PRIORITY (single source of truth) so the priority
            # the model is told never drifts from the priority the builder enforces.
            section_priority=[s.value for s in sorted(_SOURCE_PRIORITY, key=_SOURCE_PRIORITY.get)],
        )
        if context.live_data:
            sets = sorted({i.data.get("entity_set") for i in context.live_data if i.data.get("entity_set")})
            add("live_data_available", entity_sets=sets)
        if context.metadata:
            entities = sorted({
                i.data.get("entity") for i in context.metadata
                if isinstance(i.data, dict) and i.data.get("entity")
            })
            add("metadata_available", entities=entities)
        if context.tools:
            add("tools_available", tool_keys=sorted({i.ref for i in context.tools if i.ref}))
        return items

    # ── budget ──────────────────────────────────────────────────────────────────

    def _apply_budget(
        self,
        sections: Dict[ContextSection, List[ContextItem]],
        s: ContextBuilderSettings,
    ) -> Tuple[Dict[ContextSection, List[ContextItem]], int]:
        discarded = 0

        # 1) Per-section caps (items are already ranked best-first).
        for section, items in sections.items():
            cap = s.cap_for(section)
            if cap is None:
                continue
            kept: List[ContextItem] = []
            used = 0
            for it in items:
                if used + it.token_estimate <= cap:
                    kept.append(it)
                    used += it.token_estimate
                else:
                    discarded += 1
            sections[section] = kept

        # 2) Global ceiling — trim lowest-priority sections / lowest-ranked items.
        total = sum(it.token_estimate for items in sections.values() for it in items)
        if total > s.max_tokens:
            for section in _TRIM_ORDER:
                items = sections.get(section, [])
                while total > s.max_tokens and items:
                    removed = items.pop()          # lowest-ranked in this section
                    total -= removed.token_estimate
                    discarded += 1
                if total <= s.max_tokens:
                    break

        return sections, discarded
