# Conversational agent using LangChain

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.config import get_settings
import logging
import time
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatAgent:
    """Conversational AI agent powered by LangChain"""

    def __init__(self):
        model_name = getattr(settings, 'openai_model', 'gpt-3.5-turbo')
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=settings.openai_api_key,
            max_tokens=1000,
            request_timeout=30
        )
        logger.info(f"LLM initialized: {model_name}")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

        self.total_requests = 0
        self.last_request_time = None
        logger.info("ChatAgent initialized")

    def _get_system_prompt(self) -> str:
        return """You are an intelligent AI assistant specialized in SAP Business Technology Platform (BTP).

Your expertise includes:
- SAP BTP core concepts and architecture
- CAP (Cloud Application Programming) Model
- HANA Cloud database
- Cloud Foundry deployment
- Fiori applications
- OData services
- Authentication and authorization
- Integration patterns

Your communication style:
- Clear and concise explanations
- Use code examples when helpful
- Break down complex topics step-by-step
- Acknowledge when you're not certain about something
- Focus on practical, actionable advice

When helping with code:
- Explain what the code does
- Highlight best practices
- Suggest improvements when relevant
- Consider SAP-specific patterns and conventions"""

    async def get_response(self, message: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        start_time = time.time()
        self.total_requests += 1
        self.last_request_time = start_time

        try:
            logger.info(f"Processing request #{self.total_requests}: {message[:50]}...")

            # Format conversation history
            formatted_history = []
            if history:
                for msg in history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        formatted_history.append(HumanMessage(content=content))
                    elif role == "assistant":
                        formatted_history.append(AIMessage(content=content))
                    elif role == "system":
                        formatted_history.append(SystemMessage(content=content))

            response = await self.chain.ainvoke({"input": message, "history": formatted_history})
            response_time = time.time() - start_time

            logger.info(f"Response generated in {response_time:.2f}s")

            return {
                "response": response,
                "response_time": response_time,
                "model": self.llm.model_name,
                "total_requests": self.total_requests
            }

        except Exception as e:
            logger.error(f"Error in ChatAgent.get_response: {e}", exc_info=True)
            raise

    async def stream_response(self, message: str, history: List[Dict[str, str]] = None):
        self.total_requests += 1
        formatted_history = []
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    formatted_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    formatted_history.append(AIMessage(content=content))
                elif role == "system":
                    formatted_history.append(SystemMessage(content=content))

        async for chunk in self.chain.astream({"input": message, "history": formatted_history}):
            yield chunk

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_type": "chat",
            "status": "healthy",
            "model": self.llm.model_name,
            "total_requests": self.total_requests,
            "last_request_time": self.last_request_time
        }