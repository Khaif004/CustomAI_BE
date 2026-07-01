"""PlannerService — the read-only orchestrator/facade for the Planner layer.

Pipeline (NO LLM, NO response generation, NO mutation):

    message ─▶ PlanSignals.build (lexical + context signals)
            ─▶ IntentClassifier.classify (rule-based scoring)
            ─▶ EntityResolver.resolve   (in-memory registry; always — pure)
            ─▶ ToolResolver.resolve     (Tool Registry; only when TOOL_EXECUTION
                                          is the top/near-tie intent and a DB
                                          session is available)
            ─▶ post-resolution confidence adjustment (single bounded pass)
            ─▶ retrievalSources mapping + requiresLiveData rule
            ─▶ PlannerResult

The request-scoped AsyncSession is passed into `analyze`; the components
themselves are stateless and injected via the constructor (DI).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.models.planner import Intent, PlannerResult, RetrievalSource
from app.services.planner import text_signals as ts
from app.services.planner.entity_resolver import EntityResolver, ResolvedEntity
from app.services.planner.intent_classifier import IntentClassifier, IntentScore
from app.services.planner.tool_resolver import ResolvedTool, ToolResolver

# Intents that warrant attempting tool resolution even if not strictly the top.
_TOOL_NEAR_TIE = 0.15

# Base retrieval-source map per intent (ordered, most-authoritative first).
_BASE_SOURCES: Dict[Intent, List[RetrievalSource]] = {
    Intent.DATA_QUERY: [RetrievalSource.METADATA, RetrievalSource.PGVECTOR, RetrievalSource.FULL_TEXT_SEARCH],
    Intent.TOOL_EXECUTION: [RetrievalSource.TOOL_REGISTRY, RetrievalSource.METADATA],
    Intent.KNOWLEDGE: [RetrievalSource.PGVECTOR, RetrievalSource.DOCUMENTATION, RetrievalSource.FULL_TEXT_SEARCH],
    Intent.SCHEMA: [RetrievalSource.METADATA, RetrievalSource.PGVECTOR],
    Intent.NAVIGATION: [RetrievalSource.METADATA, RetrievalSource.DOCUMENTATION],
    Intent.DOCUMENTATION: [RetrievalSource.DOCUMENTATION, RetrievalSource.PGVECTOR],
    Intent.CODE_INTELLIGENCE: [RetrievalSource.CODE_SUMMARIES, RetrievalSource.PGVECTOR],
    Intent.GENERAL_CHAT: [],
}

_UI_CONTEXT_INTENTS = {Intent.DATA_QUERY, Intent.TOOL_EXECUTION, Intent.SCHEMA, Intent.NAVIGATION}
_MEMORY_INTENTS = {Intent.KNOWLEDGE, Intent.GENERAL_CHAT}
_HIGH_ENTITY_SOURCES = {"exact", "alias", "fiori", "compound"}


class PlannerService:
    def __init__(
        self,
        intent_classifier: IntentClassifier,
        entity_resolver: EntityResolver,
        tool_resolver: ToolResolver,
    ):
        self._classifier = intent_classifier
        self._entities = entity_resolver
        self._tools = tool_resolver

    async def analyze(
        self,
        message: str,
        *,
        app_id: Optional[str] = None,
        fiori_context: Optional[Dict] = None,
        session=None,
    ) -> PlannerResult:
        signals = ts.PlanSignals.build(message, app_id=app_id, fiori_context=fiori_context)

        intent_score: IntentScore = self._classifier.classify(signals)
        scores = dict(intent_score.scores)

        # Entity resolution is pure/in-memory → always run (informs sources too).
        entity: ResolvedEntity = self._entities.resolve(signals, app_id, fiori_context)

        # Tool resolution only when TOOL_EXECUTION is the top or a near-tie, and a
        # DB session is available — avoids needless DB hits for read intents.
        tool = ResolvedTool()
        te_score = scores.get(Intent.TOOL_EXECUTION, 0.0)
        top_score = max(scores.values()) if scores else 0.0
        should_resolve_tool = (
            session is not None
            and bool(app_id)
            and (intent_score.intent == Intent.TOOL_EXECUTION or te_score >= top_score - _TOOL_NEAR_TIE)
            and te_score > 0.0
        )
        if should_resolve_tool:
            tool = await self._tools.resolve(
                session, signals, app_id, entity.name, fiori_context
            )

        # ── Bounded post-resolution adjustment (still deterministic, no LLM) ──
        if tool.tool_key:
            scores[Intent.TOOL_EXECUTION] = min(1.0, scores.get(Intent.TOOL_EXECUTION, 0.0) + ts.STRONG)
        if entity.name and entity.source in _HIGH_ENTITY_SOURCES:
            for it in (Intent.DATA_QUERY, Intent.SCHEMA):
                if scores.get(it, 0.0) > 0.0:
                    scores[it] = min(1.0, scores[it] + ts.WEAK)

        final = self._classifier.finalize(scores)
        intent = final.intent
        confidence = final.confidence

        # Only surface a tool/missing-params when the final intent is TOOL_EXECUTION.
        result_tool = tool.tool_key if (intent == Intent.TOOL_EXECUTION and tool.tool_key) else None
        missing = tool.missing_parameters if result_tool else []

        requires_live = self._requires_live_data(intent, signals, entity)
        sources = self._retrieval_sources(intent, signals, requires_live)

        return PlannerResult(
            intent=intent,
            confidence=confidence,
            application=app_id or None,
            entity=entity.name,
            tool=result_tool,
            retrieval_sources=sources,
            requires_live_data=requires_live,
            missing_parameters=missing,
        )

    # ── rules ────────────────────────────────────────────────────────────────

    @staticmethod
    def _requires_live_data(
        intent: Intent, signals: ts.PlanSignals, entity: ResolvedEntity
    ) -> bool:
        """Live OData is needed only to READ current data for an app-scoped,
        entity-targeted DATA_QUERY (or a DOCUMENTATION report over live data).
        TOOL_EXECUTION/SCHEMA/KNOWLEDGE/NAVIGATION/CODE/GENERAL → False."""
        if not signals.has_app_context:
            return False
        if intent in (Intent.DATA_QUERY, Intent.DOCUMENTATION):
            return entity.name is not None
        return False

    @staticmethod
    def _retrieval_sources(
        intent: Intent, signals: ts.PlanSignals, requires_live: bool
    ) -> List[RetrievalSource]:
        ordered: List[RetrievalSource] = []

        if requires_live:
            ordered.append(RetrievalSource.LIVE_ODATA)  # most authoritative first

        ordered.extend(_BASE_SOURCES.get(intent, []))

        if intent in _UI_CONTEXT_INTENTS and (signals.has_entity_data or signals.has_service_url):
            ordered.append(RetrievalSource.UI_CONTEXT)
        if intent in _MEMORY_INTENTS:
            ordered.append(RetrievalSource.CONVERSATION_MEMORY)

        # De-dupe, preserve order.
        seen = set()
        deduped: List[RetrievalSource] = []
        for s in ordered:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped
