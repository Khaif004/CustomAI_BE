"""/api/planner — read-only Planner endpoint.

A NEW, additive router (own prefix). It does NOT modify chat.py, streaming, auth,
or any SDK endpoint. The Planner analyzes a message and returns an execution plan;
it never generates a response and never calls the LLM.

DI:
  * `planner` — the process-wide stateless PlannerService (Depends(get_planner_service)).
  * `db` — a SOFT AsyncSession (Depends(get_optional_db)); ``None`` when Neon is
           unconfigured, in which case tool resolution gracefully no-ops while
           intent + entity resolution still work.
  * `current_user` — reuses the existing auth dependency unchanged.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.db.session import get_optional_db
from app.models.planner import PlannerAnalyzeRequest, PlannerResult
from app.services.planner import get_planner_service
from app.services.planner.planner_service import PlannerService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/planner", tags=["planner"])


@router.post("/analyze", response_model=PlannerResult)
async def analyze(
    request: PlannerAnalyzeRequest,
    db: Optional[AsyncSession] = Depends(get_optional_db),
    planner: PlannerService = Depends(get_planner_service),
    current_user=Depends(get_current_user),
) -> PlannerResult:
    """Analyze a user message and return its execution plan (no response text)."""
    plan = await planner.analyze(
        request.message,
        app_id=request.app_id,
        fiori_context=request.fiori_context,
        session=db,
    )
    logger.info(
        "[planner] app_id=%s intent=%s conf=%.2f entity=%s tool=%s live=%s",
        request.app_id, plan.intent, plan.confidence, plan.entity,
        plan.tool, plan.requires_live_data,
    )
    return plan
