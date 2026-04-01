# Supervisor/Router Agent - routes queries to specialized agents

import logging
import json
from typing import Dict, Any, Optional, List
from app.agents.base_agent import BaseAgent
from app.models.chat import AgentType
from app.tools.registry import get_tools_registry
from app.knowledge.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Routes user queries to appropriate specialized agents"""

    def __init__(self):
        super().__init__(
            agent_type=AgentType.SUPERVISOR,
            name="Supervisor Agent",
            description="Routes queries to specialized agents and orchestrates responses",
            system_prompt="""You are the Supervisor Agent, responsible for orchestrating other specialized agents.

Analyze the user's query and route to: developer_helper (code/architecture), data_analyst (data/queries), architect (system design), or documentation (docs).

Always respond in JSON format with:
{
    "intent": "classified intent",
    "agents": ["recommended agent(s)"],
    "reasoning": "why these agents",
    "follow_up_actions": ["action 1", "action 2"]
}
"""
        )

    async def _process_query_internal(self, query: str, context: Optional[Dict[str, Any]] = None,
                                       conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        try:
            routing_decision = self._classify_intent(query.lower())
            kb = get_knowledge_base()
            knowledge_context = kb.search(query, k=3)

            return {
                "status": "success",
                "agent": self.agent_type.value,
                "routing_decision": routing_decision,
                "selected_agents": routing_decision["agents"],
                "knowledge_context": knowledge_context,
                "reasoning": routing_decision.get("reasoning", ""),
                "suggested_follow_up": routing_decision.get("follow_up_actions", [])
            }
        except Exception as e:
            logger.error(f"Supervisor routing error: {e}")
            return {"status": "error", "agent": self.agent_type.value, "error": str(e)}

    def _classify_intent(self, query_lower: str) -> Dict[str, Any]:
        """Classify query intent and recommend agents"""
        agents = []
        intent = "general"

        if any(kw in query_lower for kw in ["code", "function", "class", "method", "algorithm", "pattern", "architecture", "explain", "how to", "why", "refactor"]):
            agents.append("developer_helper")
            intent = "code_analysis"

        if any(kw in query_lower for kw in ["data", "query", "sql", "database", "blend", "table", "report", "analytics", "statistics", "summary"]):
            agents.append("data_analyst")
            intent = "data_analysis"

        if any(kw in query_lower for kw in ["architecture", "design", "integration", "system", "component", "layer", "service", "microservice", "deployment"]):
            agents.append("architect")
            intent = "system_design"

        if any(kw in query_lower for kw in ["document", "generate", "diagram", "flow", "visualization", "guide", "manual", "readme", "api"]):
            agents.append("documentation")
            intent = "documentation"

        if not agents:
            agents = ["developer_helper"]
            intent = "unknown"

        return {
            "intent": intent,
            "agents": agents,
            "reasoning": f"Query classified as {intent}, routing to {', '.join(agents)}",
            "follow_up_actions": ["search_knowledge_base", "collect_context"]
        }

    def get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        return {
            "developer_helper": {"name": "Developer Helper", "description": "Explains code, architecture, and design patterns"},
            "data_analyst": {"name": "Data Analyst", "description": "Provides data analysis, queries, and summaries"},
            "architect": {"name": "Architect", "description": "Explains system design and integrations"},
            "documentation": {"name": "Documentation", "description": "Generates documentation and diagrams"}
        }
