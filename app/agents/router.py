"""AgentRouter — dispatches queries to the correct agent based on context.

Routing logic
─────────────
  No app_id AND no fiori_context               →  GlobalChatAgent    (general-purpose)
  app_id present AND navigation intent         →  NavigationAgent    (rule-based, no LLM)
  app_id OR fiori_context (all other queries)  →  AppContextAgent    (schema / OData-aware)

Adding more agents: add the instance to __init__ and extend _pick_agent().
"""

import logging
from typing import Any, Dict, List, Optional

from app.agents.navigation_agent import NavigationAgent

logger = logging.getLogger(__name__)


class AgentRouter:
    """Multi-agent router that selects the right agent for each request."""

    def __init__(self, global_agent, app_agent):
        self.global_agent = global_agent    # GlobalChatAgent — general purpose
        self.app_agent    = app_agent       # SAPAICoreAgent / ChatAgent — app-aware
        self.nav_agent    = NavigationAgent()
        logger.info(
            "AgentRouter initialised  global=%s  app=%s  nav=NavigationAgent",
            type(global_agent).__name__,
            type(app_agent).__name__,
        )

    # ── Routing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _is_app_context(app_id: Optional[str], fiori_context: Optional[Dict]) -> bool:
        """Return True when the request carries an embedded-app context."""
        return bool(app_id) or bool(fiori_context)

    def _pick_agent(
        self,
        message: str,
        app_id: Optional[str],
        fiori_context: Optional[Dict],
        raw_message: Optional[str] = None,
    ):
        if self._is_app_context(app_id, fiori_context):
            # Navigation intents that carry a real app_id go to the NavigationAgent
            # (no LLM needed — rule-based extraction is fast and reliable).

            intent_text = raw_message or message
            if app_id and NavigationAgent.is_navigation_intent(intent_text):
                logger.debug("Routing → NavigationAgent (app_id=%s msg=%r)", app_id, intent_text[:60])
                return self.nav_agent
            logger.debug("Routing → AppContextAgent (app_id=%s)", app_id)
            return self.app_agent
        logger.debug("Routing → GlobalChatAgent")
        return self.global_agent

    # ── Public interface (mirrors the ChatAgent / SAPAICoreAgent interface) ──

    async def get_response(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        app_id: Optional[str] = None,
        fiori_context: Optional[Dict] = None,
        odata_token: Optional[str] = None,
        user_id: Optional[str] = None,
        raw_message: Optional[str] = None,
        backend_url: Optional[str] = None,
        prepared_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        agent = self._pick_agent(message, app_id, fiori_context, raw_message=raw_message)
        return await agent.get_response(
            message=message,
            history=history,
            app_id=app_id,
            fiori_context=fiori_context,
            odata_token=odata_token,
            user_id=user_id,
            raw_message=raw_message,
            backend_url=backend_url,
            prepared_context=prepared_context,
        )

    async def stream_response(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        app_id: Optional[str] = None,
        fiori_context: Optional[Dict] = None,
        odata_token: Optional[str] = None,
        user_id: Optional[str] = None,
        raw_message: Optional[str] = None,
        backend_url: Optional[str] = None,
        prepared_context: Optional[str] = None,
    ):
        agent = self._pick_agent(message, app_id, fiori_context, raw_message=raw_message)
        async for chunk in agent.stream_response(
            message=message,
            history=history,
            app_id=app_id,
            fiori_context=fiori_context,
            odata_token=odata_token,
            user_id=user_id,
            raw_message=raw_message,
            backend_url=backend_url,
            prepared_context=prepared_context,
        ):
            yield chunk

    def get_status(self) -> Dict[str, Any]:
        """Return a flat status dict compatible with AgentStatus response model."""
        try:
            app_status = self.app_agent.get_status()
        except Exception:
            app_status = {"status": "unknown"}
        try:
            global_status = self.global_agent.get_status()
        except Exception:
            global_status = {"status": "unknown"}

        return {
            "agent_type": "router",
            "status": "healthy",
            "model": app_status.get("model", global_status.get("model", "unknown")),
            "total_requests": (
                app_status.get("total_requests", 0)
                + global_status.get("total_requests", 0)
            ),
        }

    def get_detailed_status(self) -> Dict[str, Any]:
        """Extended status including per-agent breakdown (for /health or debug endpoints)."""
        flat = self.get_status()
        try:
            flat["agents"] = {
                "global": self.global_agent.get_status(),
                "app_context": self.app_agent.get_status(),
            }
        except Exception:
            pass
        return flat
