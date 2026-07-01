"""/api/context — read-only endpoint that runs Planner → Orchestrator → Context Builder.

A NEW, additive router (own prefix). It does NOT modify chat.py, streaming, auth,
or any SDK endpoint. It returns an `LLMContext` (purely structured data — no
prompts, no LLM, no SAP AI Core).

The Context Builder itself consumes ONLY the RetrievalContext; this endpoint is
just the wiring that produces that RetrievalContext (via the existing Planner +
Retrieval Orchestrator) and hands it to the builder. The builder never sees the
Planner, retrievers, or any model.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.db.session import get_optional_db
from app.models.llm_context import LLMContext
from app.models.planner import PlannerAnalyzeRequest
from app.services.context_builder import ContextBuilder, get_context_builder
from app.services.planner import get_planner_service
from app.services.planner.planner_service import PlannerService
from app.services.retrieval import RetrievalOrchestrator, get_retrieval_orchestrator
from app.services.retrieval.models import RetrievalRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/context", tags=["context"])


def _derive_user_id(current_user) -> Optional[str]:
    if not isinstance(current_user, dict):
        return None
    return current_user.get("sub") or current_user.get("user_name") or current_user.get("email")


@router.post("/build", response_model=LLMContext)
async def build_context(
    request: PlannerAnalyzeRequest,
    db: Optional[AsyncSession] = Depends(get_optional_db),
    planner: PlannerService = Depends(get_planner_service),
    orchestrator: RetrievalOrchestrator = Depends(get_retrieval_orchestrator),
    builder: ContextBuilder = Depends(get_context_builder),
    current_user=Depends(get_current_user),
) -> LLMContext:
    """Plan → retrieve → build a structured LLMContext (no prompt, no LLM)."""
    plan = await planner.analyze(
        request.message,
        app_id=request.app_id,
        fiori_context=request.fiori_context,
        session=db,
    )
    fc = request.fiori_context or {}
    retrieval_request = RetrievalRequest(
        message=request.message,
        plan=plan,
        app_id=request.app_id,
        fiori_context=request.fiori_context,
        odata_token=fc.get("odata_token") or fc.get("odataToken"),
        user_id=_derive_user_id(current_user),
        session=db,
    )
    retrieval_context = await orchestrator.retrieve(retrieval_request)
    # The builder consumes ONLY the RetrievalContext.
    llm_context = builder.build(retrieval_context)
    logger.info(
        "[context] app_id=%s tokens=%s retrievers=%s discarded=%s duplicates=%s",
        request.app_id, llm_context.statistics.token_estimate,
        llm_context.statistics.retrievers_used,
        llm_context.statistics.documents_discarded,
        llm_context.statistics.duplicate_count,
    )
    return llm_context
