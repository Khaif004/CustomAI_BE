# Conversational agent using LangChain

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.config import get_settings
import logging
import time
from typing import List, Dict, Any, Optional

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
            request_timeout=30,
            streaming=True
        )

        logger.info(f"LLM initialized: {model_name}")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

        self.total_requests = 0
        self.last_request_time = None

        logger.info("ChatAgent initialized")

    def _format_history(self, history):
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

        return formatted_history

    def _build_system_prompt(self, rag_context: Optional[str] = None, app_id: Optional[str] = None) -> str:
        base = """You are an intelligent AI assistant specialized in SAP Business Technology Platform (BTP).

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
- Consider SAP-specific patterns and conventions

When including links:
- Always use markdown link format: [Descriptive Title](url)
- The link text must be a short, meaningful human-readable title — never paste the raw URL as the label
- Good examples: [Virat Kohli – Wikipedia](https://en.wikipedia.org/wiki/Virat_Kohli), [SAP BTP Documentation](https://help.sap.com/docs/btp)
- Bad examples: [https://en.wikipedia.org/wiki/Virat_Kohli](https://en.wikipedia.org/wiki/Virat_Kohli)"""

        if rag_context:
            scoped = (
                f"\n\nYou are currently assisting a user inside the '{app_id}' application.\n"
                "Use the following retrieved context from that application to answer accurately.\n"
                "If the question is general, answer normally. If it's about this application's data "
                "or entities, prioritise the context below.\n\n"
                f"{rag_context}"
            )
            return base + scoped

        return base

    async def get_response(
        self,
        message: str,
        history: List[Dict[str, str]] = None,
        app_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        start_time = time.time()
        self.total_requests += 1
        self.last_request_time = start_time

        try:
            logger.info(f"Processing request #{self.total_requests}: {message[:50]}...")

            rag_context = await self._fetch_rag_context(message, app_id)
            system_prompt = self._build_system_prompt(rag_context, app_id)
            formatted_history = self._format_history(history)

            response = await self.chain.ainvoke({
                "system_prompt": system_prompt,
                "input": message,
                "history": formatted_history,
            })

            response_time = time.time() - start_time
            logger.info(f"Response generated in {response_time:.2f}s")

            return {
                "response": response,
                "response_time": response_time,
                "model": self.llm.model_name,
                "total_requests": self.total_requests,
            }

        except Exception as e:
            logger.error(f"Error in ChatAgent.get_response: {e}", exc_info=True)
            raise

    async def stream_response(
        self,
        message: str,
        history: List[Dict[str, str]] = None,
        app_id: Optional[str] = None,
    ):
        self.total_requests += 1
        self.last_request_time = time.time()

        try:
            rag_context = await self._fetch_rag_context(message, app_id)
            system_prompt = self._build_system_prompt(rag_context, app_id)
            formatted_history = self._format_history(history)

            async for chunk in self.chain.astream({
                "system_prompt": system_prompt,
                "input": message,
                "history": formatted_history,
            }):
                if isinstance(chunk, str):
                    text = chunk
                elif hasattr(chunk, "content"):
                    text = chunk.content or ""
                else:
                    text = str(chunk)

                if text:
                    yield text

        except Exception as e:
            logger.error(f"Error in stream_response: {e}", exc_info=True)
            raise

    async def _fetch_rag_context(self, message: str, app_id: Optional[str]) -> Optional[str]:
        """Retrieve relevant chunks from the vector store for the given query + app_id."""
        if not app_id:
            return None
        try:
            from app.knowledge.knowledge_base import get_knowledge_base
            kb = get_knowledge_base()
            ctx = kb.search_with_app_context(query=message, app_id=app_id)
            return ctx if ctx else None
        except Exception as e:
            logger.warning(f"RAG context fetch failed for app '{app_id}': {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_type": "chat",
            "status": "healthy",
            "model": self.llm.model_name,
            "total_requests": self.total_requests,
            "last_request_time": self.last_request_time,
        }



class ChatAgent:
    """Conversational AI agent powered by LangChain"""

    def __init__(self):
        model_name = getattr(settings, 'openai_model', 'gpt-3.5-turbo')

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=settings.openai_api_key,
            max_tokens=1000,
            request_timeout=30,
            streaming=True
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

    def _format_history(self, history):
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

        return formatted_history

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
- Consider SAP-specific patterns and conventions

When including links:
- Always use markdown link format: [Descriptive Title](url)
- The link text must be a short, meaningful human-readable title — never paste the raw URL as the label
- Good examples: [Virat Kohli – Wikipedia](https://en.wikipedia.org/wiki/Virat_Kohli), [SAP BTP Documentation](https://help.sap.com/docs/btp)
- Bad examples: [https://en.wikipedia.org/wiki/Virat_Kohli](https://en.wikipedia.org/wiki/Virat_Kohli)"""

    async def get_response(
        self,
        message: str,
        history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:

        start_time = time.time()

        self.total_requests += 1
        self.last_request_time = start_time

        try:
            logger.info(
                f"Processing request #{self.total_requests}: "
                f"{message[:50]}..."
            )

            formatted_history = self._format_history(history)

            response = await self.chain.ainvoke({
                "input": message,
                "history": formatted_history
            })

            response_time = time.time() - start_time

            logger.info(
                f"Response generated in {response_time:.2f}s"
            )

            return {
                "response": response,
                "response_time": response_time,
                "model": self.llm.model_name,
                "total_requests": self.total_requests
            }

        except Exception as e:
            logger.error(
                f"Error in ChatAgent.get_response: {e}",
                exc_info=True
            )
            raise

    async def stream_response(
        self,
        message: str,
        history: List[Dict[str, str]] = None
    ):
        self.total_requests += 1
        self.last_request_time = time.time()

        try:
            formatted_history = self._format_history(history)

            async for chunk in self.chain.astream({
                "input": message,
                "history": formatted_history
            }):
                if isinstance(chunk, str):
                    text = chunk
                elif hasattr(chunk, "content"):
                    text = chunk.content or ""
                else:
                    text = str(chunk)

                if text:
                    yield text

        except Exception as e:
            logger.error(
                f"Error in stream_response: {e}",
                exc_info=True
            )
            raise

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_type": "chat",
            "status": "healthy",
            "model": self.llm.model_name,
            "total_requests": self.total_requests,
            "last_request_time": self.last_request_time
        }