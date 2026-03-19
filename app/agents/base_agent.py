import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Awaitable
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import get_settings
from app.models.chat import AgentType
from app.tools.registry import get_tools_registry

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(
        self,
        agent_type: AgentType,
        name: str,
        description: str,
        system_prompt: str,
        model_name: Optional[str] = None
    ):
        """
        Initialize base agent
        
        Args:
            agent_type: Type of agent
            name: Agent name
            description: Agent description
            system_prompt: System prompt for the agent
            model_name: LLM model to use
        """
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.model_name = model_name or settings.openai_model
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            openai_api_key=settings.openai_api_key
        )
        
        # Get available tools
        tools_registry = get_tools_registry()
        self.available_tools = tools_registry.get_tools_for_agent(agent_type.value)
        
        # Statistics
        self.total_calls = 0
        self.total_tokens_used = 0
        self.last_used = None
        self.error_count = 0
        
        logger.info(f"Initialized {self.name} agent with {len(self.available_tools)} tools")
    
    def _build_prompt(self) -> ChatPromptTemplate:
        """Build prompt template for agent"""
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Process a query (abstract method - must be implemented by subclasses)
        
        Args:
            query: User query
            context: Optional context
            conversation_history: Optional conversation history
            
        Returns:
            Agent response
        """
        try:
            self.total_calls += 1
            self.last_used = datetime.utcnow()
            
            logger.debug(f"{self.name} processing query: {query[:100]}")
            
            # Build response
            response = await self._process_query_internal(
                query,
                context,
                conversation_history
            )
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in {self.name}: {str(e)}")
            return {
                "status": "error",
                "agent": self.agent_type.value,
                "error": str(e),
                "message": f"Error processing query: {str(e)}"
            }
    
    @abstractmethod
    async def _process_query_internal(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """Internal query processing - must be implemented by subclasses"""
        pass
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_type": self.agent_type.value,
            "name": self.name,
            "total_calls": self.total_calls,
            "total_tokens_used": self.total_tokens_used,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "error_count": self.error_count,
            "is_healthy": self.error_count < 5,
            "available_tools": len(self.available_tools)
        }
