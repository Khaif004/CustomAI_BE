"""ChatPipelineService — runs the existing pipeline for one chat turn:

    ConversationContext
      → Planner.analyze                 (intent / sources / requires_live_data)
      → RetrievalOrchestrator.retrieve  (RetrievalContext)   [Chat NEVER calls retrievers directly]
      → ContextBuilder.build            (LLMContext)         [RetrievalContext goes straight in]
      → render_llm_context              (prepared_context string for the existing agents)

It REUSES the existing `@lru_cache` singletons (`get_planner_service`,
`get_retrieval_orchestrator`, `get_context_builder`) — nothing is re-implemented.
Every stage is timed and logged with structured fields (execution time,
retrievers used, planner confidence, token estimate, errors). It returns a
`PipelineOutput`, or `None` when context could not be built (the chat layer then
falls back to the existing flow).

This service contains NO action execution, navigation, conversation-memory, or
prompt-building logic — those are future phases.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from app.models.conversation_context import ConversationContext
from app.models.llm_context import LLMContext
from app.services.chat_context.renderer import render_llm_context
from app.services.context_builder import get_context_builder
from app.services.planner import get_planner_service
from app.services.retrieval import get_retrieval_orchestrator
from app.services.retrieval.models import RetrievalRequest

logger = logging.getLogger("chat.pipeline")


@dataclass
class PipelineOutput:
    llm_context: LLMContext
    prepared_context: str          # rendered text fed to the agents (may be "")
    intent: str
    confidence: float
    requires_live_data: bool
    token_estimate: int
    retrievers_used: list
    total_ms: float


class ChatPipelineService:
    """Stateless orchestrator over the existing pipeline singletons."""

    def __init__(self, planner=None, orchestrator=None, builder=None):
        self._planner = planner or get_planner_service()
        self._orchestrator = orchestrator or get_retrieval_orchestrator()
        self._builder = builder or get_context_builder()

    async def run(self, cc: ConversationContext, *, session=None) -> Optional[PipelineOutput]:
        log = {"channel": cc.channel.value, "app_id": cc.app_id, "session_id": cc.session_id}

        # ── Stage 1: Planner (analyzes message + ConversationContext) ───────────
        t = time.perf_counter()
        plan = await self._planner.analyze(
            cc.message, app_id=cc.app_id, fiori_context=cc.fiori_context, session=session,
        )
        dt_plan = (time.perf_counter() - t) * 1000
        logger.info(
            "[pipeline.planner] %s intent=%s confidence=%.3f sources=%s requires_live=%s "
            "entity=%s tool=%s ms=%.1f",
            log, plan.intent.value, plan.confidence,
            [s.value for s in plan.retrieval_sources], plan.requires_live_data,
            plan.entity, plan.tool, dt_plan,
        )

        # ── Stage 2: Retrieval Orchestrator (driven by the plan) ────────────────
        t = time.perf_counter()
        rreq = RetrievalRequest(
            message=cc.message, plan=plan, app_id=cc.app_id,
            fiori_context=cc.fiori_context, odata_token=cc.odata_token,
            user_id=cc.user_id, session=session,
        )
        rctx = await self._orchestrator.retrieve(rreq)
        dt_ret = (time.perf_counter() - t) * 1000
        logger.info(
            "[pipeline.retrieval] %s sources_run=%s errors=%s ms=%.1f",
            log, [s.value for s in rctx.sources_run], list(rctx.errors.keys()), dt_ret,
        )

        # ── Stage 3: Context Builder (RetrievalContext passed directly in) ──────
        t = time.perf_counter()
        llm_ctx: LLMContext = self._builder.build(rctx)
        dt_build = (time.perf_counter() - t) * 1000
        st = llm_ctx.statistics
        logger.info(
            "[pipeline.context_builder] %s token_estimate=%d retrievers_used=%s discarded=%d "
            "duplicates=%d compression_ratio=%.3f ms=%.1f",
            log, st.token_estimate, st.retrievers_used, st.documents_discarded,
            st.duplicate_count, st.compression_ratio, dt_build,
        )

        # ── Stage 4: Prompt-generation adapter (TEMPORARY renderer) ─────────────
        t = time.perf_counter()
        prepared = render_llm_context(llm_ctx)
        dt_render = (time.perf_counter() - t) * 1000
        total_ms = dt_plan + dt_ret + dt_build + dt_render
        logger.info(
            "[pipeline.prompt_gen] %s prepared_chars=%d ms=%.1f total_ms=%.1f",
            log, len(prepared), dt_render, total_ms,
        )

        return PipelineOutput(
            llm_context=llm_ctx,
            prepared_context=prepared,
            intent=plan.intent.value,
            confidence=plan.confidence,
            requires_live_data=plan.requires_live_data,
            token_estimate=st.token_estimate,
            retrievers_used=list(st.retrievers_used),
            total_ms=round(total_ms, 1),
        )


_pipeline_singleton: Optional[ChatPipelineService] = None


def get_chat_pipeline() -> ChatPipelineService:
    """Process-wide stateless ChatPipelineService (lazy)."""
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = ChatPipelineService()
    return _pipeline_singleton
