"""/api/retrieval — read-only endpoint that runs Planner → Retrieval Orchestrator.

A NEW, additive router (own prefix). It does NOT modify chat.py, streaming, auth,
or any SDK endpoint, and the Chat API never calls retrievers directly — only this
route invokes the orchestrator. It returns a `RetrievalContext` (no prompts, no
response generation, no SAP AI Core).

Pipeline: PlannerAnalyzeRequest → get_planner_service().analyze() → PlannerResult
→ get_retrieval_orchestrator().retrieve() → RetrievalContext.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.db.session import get_optional_db
from app.models.planner import PlannerAnalyzeRequest
from app.services.planner import get_planner_service
from app.services.planner.planner_service import PlannerService
from app.services.retrieval import RetrievalOrchestrator, get_retrieval_orchestrator
from app.services.retrieval.models import RetrievalContext, RetrievalRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


def _derive_user_id(current_user) -> Optional[str]:
    if not isinstance(current_user, dict):
        return None
    return current_user.get("sub") or current_user.get("user_name") or current_user.get("email")


@router.post("/context", response_model=RetrievalContext)
async def build_retrieval_context(
    request: PlannerAnalyzeRequest,
    db: Optional[AsyncSession] = Depends(get_optional_db),
    planner: PlannerService = Depends(get_planner_service),
    orchestrator: RetrievalOrchestrator = Depends(get_retrieval_orchestrator),
    current_user=Depends(get_current_user),
) -> RetrievalContext:
    """Plan the request, then execute the retrieval strategy into a RetrievalContext."""
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
    context = await orchestrator.retrieve(retrieval_request)
    logger.info(
        "[retrieval] app_id=%s intent=%s sources=%s errors=%s",
        request.app_id, plan.intent,
        [s.value for s in context.sources_run], list(context.errors.keys()),
    )
    return context
