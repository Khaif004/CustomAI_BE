"""
Supervisor/Router Agent

Routes user queries to appropriate specialized agents.
Analyzes query intent and selects the best agent(s) to handle it.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from app.agents.base_agent import BaseAgent
from app.models.chat import AgentType
from app.tools.registry import get_tools_registry
from app.knowledge.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent for query routing and orchestration
    
    Responsibilities:
    - Classify query intent
    - Route to appropriate agents
    - Aggregate responses
    - Handle fallback
    """
    
    def __init__(self):
        """Initialize supervisor agent"""
        system_prompt = """You are the Supervisor Agent, responsible for orchestrating other specialized agents.

Your responsibilities:
1. Analyze the user's query carefully
2. Determine the best agent(s) to handle it
3. Route to: developer_helper (code/architecture), data_analyst (data/queries), architect (system design), or documentation (docs)
4. Ensure accurate and comprehensive responses

When routing, consider:
- Query keywords and intent
- Required expertise level
- Agent availability
- Conversation context

Always respond in JSON format with:
{
    "intent": "classified intent",
    "agents": ["recommended agent(s)"],
    "reasoning": "why these agents",
    "follow_up_actions": ["action 1", "action 2"]
}
"""
        super().__init__(
            agent_type=AgentType.SUPERVISOR,
            name="Supervisor Agent",
            description="Routes queries to specialized agents and orchestrates responses",
            system_prompt=system_prompt
        )
    
    async def _process_query_internal(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Process query and determine routing
        
        Args:
            query: User query
            context: Optional context
            conversation_history: Optional conversation history
            
        Returns:
            Routing decision and recommended agents
        """
        try:
            # Extract keywords to help determine agent
            keywords_lower = query.lower()
            
            # Intent classification
            routing_decision = self._classify_intent(keywords_lower)
            
            # Search knowledge base for context
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
            logger.error(f"Supervisor routing error: {str(e)}")
            return {
                "status": "error",
                "agent": self.agent_type.value,
                "error": str(e)
            }
    
    def _classify_intent(self, query_lower: str) -> Dict[str, Any]:
        """
        Classify query intent and recommend agents
        
        Args:
            query_lower: Lowercase query for keyword matching
            
        Returns:
            Intent classification and agent recommendations
        """
        agents = []
        intent = "general"
        
        # Developer Helper keywords
        if any(kw in query_lower for kw in ["code", "function", "class", "method", "algorithm", "pattern", "architecture", "explain", "how to", "why", "refactor"]):
            agents.append("developer_helper")
            intent = "code_analysis"
        
        # Data Analyst keywords
        if any(kw in query_lower for kw in ["data", "query", "sql", "database", "blend", "table", "report", "analytics", "statistics", "summary"]):
            agents.append("data_analyst")
            intent = "data_analysis"
        
        # Architect keywords
        if any(kw in query_lower for kw in ["architecture", "design", "integration", "system", "component", "layer", "service", "microservice", "deployment"]):
            agents.append("architect")
            intent = "system_design"
        
        # Documentation keywords
        if any(kw in query_lower for kw in ["document", "generate", "diagram", "flow", "visualization", "guide", "manual", "readme", "api"]):
            agents.append("documentation")
            intent = "documentation"
        
        # If no specific agent matched, use developer helper as fallback
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
        """Get information about available agents"""
        return {
            "developer_helper": {
                "name": "Developer Helper",
                "description": "Explains code, architecture, and design patterns",
                "keywords": ["code", "function", "architecture", "pattern"]
            },
            "data_analyst": {
                "name": "Data Analyst",
                "description": "Provides data analysis, queries, and summaries",
                "keywords": ["data", "query", "analytics", "blend"]
            },
            "architect": {
                "name": "Architect",
                "description": "Explains system design and integrations",
                "keywords": ["architecture", "design", "integration", "system"]
            },
            "documentation": {
                "name": "Documentation",
                "description": "Generates documentation and diagrams",
                "keywords": ["document", "diagram", "guide", "api"]
            }
        }
