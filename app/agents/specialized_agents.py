# Specialized agent implementations

import logging
from typing import Dict, Any, Optional, List
from app.agents.base_agent import BaseAgent
from app.models.chat import AgentType
from app.knowledge.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


class DeveloperHelperAgent(BaseAgent):
    """Agent for code analysis and architecture explanation"""

    def __init__(self):
        super().__init__(
            agent_type=AgentType.DEVELOPER_HELPER,
            name="Developer Helper",
            description="Code analysis and architecture explanation",
            system_prompt="""You are the Developer Helper Agent, specialized in code and architecture analysis.

Your expertise:
- Explain source code structure and logic
- Analyze system architecture and design patterns
- Identify best practices and code improvements
- Explain relationships between components
- Provide technical guidance"""
        )

    async def _process_query_internal(self, query: str, context: Optional[Dict[str, Any]] = None,
                                       conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        try:
            kb = get_knowledge_base()
            related_docs = kb.search(query, k=5, category_filter="code")
            return {"status": "success", "agent": self.agent_type.value, "response": f"Developer analysis for: {query}",
                    "related_code": related_docs, "analysis_type": "code_architecture"}
        except Exception as e:
            logger.error(f"Developer helper error: {e}")
            return {"status": "error", "agent": self.agent_type.value, "error": str(e)}


class DataAnalystAgent(BaseAgent):
    """Agent for data analysis and queries"""

    def __init__(self):
        super().__init__(
            agent_type=AgentType.DATA_ANALYST,
            name="Data Analyst",
            description="Data analysis and query generation",
            system_prompt="""You are the Data Analyst Agent, specialized in data analysis and queries.

Your expertise:
- Generate and explain SQL queries
- Analyze data structures and relationships
- Summarize data and provide insights
- Explain data transformations and blends
- Create data profiles and reports"""
        )

    async def _process_query_internal(self, query: str, context: Optional[Dict[str, Any]] = None,
                                       conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        try:
            kb = get_knowledge_base()
            related_docs = kb.search(query, k=5, category_filter="data")
            return {"status": "success", "agent": self.agent_type.value, "response": f"Data analysis for: {query}",
                    "related_data_docs": related_docs, "analysis_type": "data_query"}
        except Exception as e:
            logger.error(f"Data analyst error: {e}")
            return {"status": "error", "agent": self.agent_type.value, "error": str(e)}


class ArchitectAgent(BaseAgent):
    """Agent for system architecture and design"""

    def __init__(self):
        super().__init__(
            agent_type=AgentType.ARCHITECT,
            name="Architect",
            description="System architecture and design",
            system_prompt="""You are the Architect Agent, specialized in system design and integration.

Your expertise:
- Explain system architecture and design
- Describe component interactions and integrations
- Analyze technology choices and trade-offs
- Design resilience and scalability patterns
- Create architecture documentation"""
        )

    async def _process_query_internal(self, query: str, context: Optional[Dict[str, Any]] = None,
                                       conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        try:
            kb = get_knowledge_base()
            related_docs = kb.search(query, k=5, category_filter="architecture")
            return {"status": "success", "agent": self.agent_type.value, "response": f"Architecture analysis for: {query}",
                    "related_architecture": related_docs, "analysis_type": "system_design"}
        except Exception as e:
            logger.error(f"Architect error: {e}")
            return {"status": "error", "agent": self.agent_type.value, "error": str(e)}


class DocumentationGeneratorAgent(BaseAgent):
    """Agent for documentation generation"""

    def __init__(self):
        super().__init__(
            agent_type=AgentType.DOCUMENTATION,
            name="Documentation Generator",
            description="Documentation and diagram generation",
            system_prompt="""You are the Documentation Generator Agent, specialized in creating documentation.

Your expertise:
- Generate comprehensive documentation
- Create architecture diagrams and flowcharts (ASCII)
- Write API documentation
- Create deployment guides
- Generate README and user guides"""
        )

    async def _process_query_internal(self, query: str, context: Optional[Dict[str, Any]] = None,
                                       conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        try:
            kb = get_knowledge_base()
            related_docs = kb.search(query, k=5)
            return {"status": "success", "agent": self.agent_type.value, "response": f"Documentation for: {query}",
                    "related_content": related_docs, "documentation_type": "generated"}
        except Exception as e:
            logger.error(f"Documentation generator error: {e}")
            return {"status": "error", "agent": self.agent_type.value, "error": str(e)}


class AgentFactory:
    """Factory for creating and caching agent instances"""

    _agents = {}

    @classmethod
    def get_agent(cls, agent_type: AgentType) -> BaseAgent:
        if agent_type not in cls._agents:
            if agent_type == AgentType.SUPERVISOR:
                from app.agents.supervisor_agent import SupervisorAgent
                cls._agents[agent_type] = SupervisorAgent()
            elif agent_type == AgentType.DEVELOPER_HELPER:
                cls._agents[agent_type] = DeveloperHelperAgent()
            elif agent_type == AgentType.DATA_ANALYST:
                cls._agents[agent_type] = DataAnalystAgent()
            elif agent_type == AgentType.ARCHITECT:
                cls._agents[agent_type] = ArchitectAgent()
            elif agent_type == AgentType.DOCUMENTATION:
                cls._agents[agent_type] = DocumentationGeneratorAgent()
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
        return cls._agents[agent_type]

    @classmethod
    def get_all_agents(cls) -> Dict[str, BaseAgent]:
        return {agent_type.value: cls.get_agent(agent_type) for agent_type in AgentType}
