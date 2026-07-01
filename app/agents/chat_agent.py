# Conversational agent using LangChain

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.config import get_settings
import aiohttp
import logging
import re
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
settings = get_settings()

# Questions that need live DB access
_COUNT_RE = re.compile(
    r'\b(how many|count|total|how much|number of|records?|rows?|entries)\b',
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r'\b(show me|list|get|fetch|give me|display|what are|find all|all the)\b',
    re.IGNORECASE,
)
_LIVE_DATA_RE = re.compile(
    r'\b(how many|count|total|how much|number of|records?|rows?|entries|'
    r'data in|stored in|show me|list|recent|latest|last \d+|find|fetch|give me)\b',
    re.IGNORECASE,
)


class ChatAgent:
    """Conversational AI agent powered by LangChain with RAG and live OData support."""

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
        formatted = []
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    formatted.append(HumanMessage(content=content))
                elif role == "assistant":
                    formatted.append(AIMessage(content=content))
                elif role == "system":
                    formatted.append(SystemMessage(content=content))
        return formatted

    def _build_system_prompt(
        self,
        rag_context: Optional[str] = None,
        app_id: Optional[str] = None,
        live_data: Optional[str] = None,
    ) -> str:
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

        additions = []

        if rag_context:
            additions.append(
                f"\n\nYou are currently assisting a user inside the '{app_id}' application.\n"
                "Use the following retrieved context from that application to answer accurately.\n"
                "If the question is general, answer normally. If it's about this application's data "
                "or entities, prioritise the context below.\n\n"
                f"{rag_context}"
            )

        if live_data:
            additions.append(
                "\n\nLIVE DATA (queried right now from the OData service — use these exact numbers):\n"
                f"{live_data}"
            )

        return base + "".join(additions)

    async def get_response(
        self,
        message: str,
        history: List[Dict[str, str]] = None,
        app_id: Optional[str] = None,
        fiori_context: Optional[Dict[str, Any]] = None,
        odata_token: Optional[str] = None,
        raw_message: Optional[str] = None,
        **_kwargs,
    ) -> Dict[str, Any]:

        start_time = time.time()
        self.total_requests += 1
        self.last_request_time = start_time

        try:
            logger.info(f"Processing request #{self.total_requests}: {message[:50]}...")

            # When the new context pipeline supplies prepared_context, use it and
            # SKIP this agent's own retrieval. None => unchanged legacy behavior.
            prepared_context = _kwargs.get("prepared_context")
            if prepared_context is not None:
                system_prompt = self._build_system_prompt(prepared_context, app_id, None)
            else:
                rag_context = await self._fetch_rag_context(message, app_id)
                live_data = await self._fetch_live_odata_counts(message, fiori_context, odata_token)
                system_prompt = self._build_system_prompt(rag_context, app_id, live_data)
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
            logger.exception(f"Error in ChatAgent.get_response: {e}")
            raise

    async def stream_response(
        self,
        message: str,
        history: List[Dict[str, str]] = None,
        app_id: Optional[str] = None,
        fiori_context: Optional[Dict[str, Any]] = None,
        odata_token: Optional[str] = None,
        raw_message: Optional[str] = None,
        **_kwargs,
    ):
        self.total_requests += 1
        self.last_request_time = time.time()

        try:
            # When the new context pipeline supplies prepared_context, use it and
            # SKIP this agent's own retrieval. None => unchanged legacy behavior.
            prepared_context = _kwargs.get("prepared_context")
            if prepared_context is not None:
                system_prompt = self._build_system_prompt(prepared_context, app_id, None)
            else:
                rag_context = await self._fetch_rag_context(message, app_id)
                live_data = await self._fetch_live_odata_counts(message, fiori_context, odata_token)
                system_prompt = self._build_system_prompt(rag_context, app_id, live_data)
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
            logger.exception(f"Error in stream_response: {e}")
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

    async def _fetch_live_odata_counts(
        self,
        message: str,
        fiori_context: Optional[Dict[str, Any]],
        odata_token: Optional[str] = None,
    ) -> Optional[str]:
        """
        When the user's message asks about record counts or data, call the OData
        service's $count endpoint for each entity and inject real numbers into the
        system prompt.

        Token priority: fiori_context.odata_token > standalone odata_token param.
        Entity names are parsed from the schema_hint that the widget sends via
        ODataProbe (lines like "Entity: EntityName" or "Entities: A, B, C").
        """
        if not fiori_context or not _LIVE_DATA_RE.search(message):
            return None

        service_url = fiori_context.get("service_url") or fiori_context.get("serviceUrl")
        token = fiori_context.get("odata_token") or odata_token
        extra = fiori_context.get("extra") or {}
        schema_hint = extra.get("schema_hint") if isinstance(extra, dict) else None

        # Make service_url absolute if it's relative (widget sends relative paths like /odata/v4/...)
        # Use page_url from context to derive the host:port
        if service_url and not service_url.startswith("http"):
            page_url = extra.get("page_url", "") if isinstance(extra, dict) else ""
            if page_url:
                try:
                    from urllib.parse import urlparse as _urlparse
                    _parsed = _urlparse(page_url)
                    service_url = f"{_parsed.scheme}://{_parsed.netloc}{service_url}"
                except Exception:
                    service_url = None
            else:
                service_url = None

        # Fallback: look up the service tool registry using app_id from fiori_context
        if not service_url:
            _app_id = fiori_context.get("app_id") or fiori_context.get("appId")
            if _app_id:
                try:
                    from app.api.apps import get_service_tool
                    _tool = get_service_tool(_app_id)
                    if _tool:
                        service_url = _tool.get("service_url")
                except Exception:
                    pass

        if not service_url or not service_url.startswith("http"):
            logger.debug("Live OData skipped — could not resolve absolute service_url")
            return None

        entities: List[str] = []
        if schema_hint:
            for line in schema_hint.splitlines():
                m = re.match(r'^Entity:\s*(\w+)', line.strip())
                if m:
                    entities.append(m.group(1))
                m2 = re.match(r'^Entities:\s*(.+)', line.strip())
                if m2:
                    entities.extend(e.strip() for e in m2.group(1).split(",") if e.strip())

        if not entities:
            return None

        headers: Dict[str, str] = {"Accept": "application/json", "OData-MaxVersion": "4.0"}
        if token:
            raw = token.replace("Bearer ", "").replace("bearer ", "")
            headers["Authorization"] = f"Bearer {raw}"

        base = service_url.rstrip("/")
        want_rows = bool(_LIST_RE.search(message))
        result_blocks: List[str] = []

        try:
            async with aiohttp.ClientSession() as session:
                for entity in entities[:8]:
                    block = await self._query_entity(
                        session, base, entity, headers, want_rows
                    )
                    if block:
                        result_blocks.append(block)
        except Exception as e:
            logger.debug(f"Live OData fetch failed: {e}")
            return None

        if not result_blocks:
            return None

        label = "Live data from OData service:" if want_rows else "Record counts from OData service:"
        return label + "\n" + "\n".join(result_blocks)

    async def _query_entity(
        self,
        session: aiohttp.ClientSession,
        base: str,
        entity: str,
        headers: Dict[str, str],
        want_rows: bool,
    ) -> Optional[str]:
        """Fetch either a $count or top-10 rows for one entity set."""
        timeout = aiohttp.ClientTimeout(total=6)
        try:
            if want_rows:
                url = f"{base}/{entity}"
                params = {"$top": "10", "$count": "true"}
                async with session.get(url, params=params, headers=headers, timeout=timeout) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    rows: List[Dict] = data.get("value", [])
                    total = data.get("@odata.count")
                    if not rows:
                        return f"  {entity}: no records found"
                    total_str = f" (total: {int(total):,})" if total is not None else ""
                    lines = [f"  {entity}{total_str}:"]
                    for row in rows:
                        # Format each row as key=value pairs, skip internal OData fields
                        pairs = ", ".join(
                            f"{k}={v}"
                            for k, v in row.items()
                            if not k.startswith("@") and v is not None
                        )
                        lines.append(f"    {{ {pairs} }}")
                    return "\n".join(lines)
            else:
                url = f"{base}/{entity}/$count"
                async with session.get(url, headers=headers, timeout=timeout) as resp:
                    if resp.status != 200:
                        return None
                    text = (await resp.text()).strip()
                    return f"  {entity}: {int(text):,} record(s)"
        except Exception:
            return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_type": "chat",
            "status": "healthy",
            "model": self.llm.model_name,
            "total_requests": self.total_requests,
            "last_request_time": self.last_request_time,
        }
