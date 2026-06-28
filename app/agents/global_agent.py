"""GlobalChatAgent — general-purpose AI assistant with no app-context restrictions."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GLOBAL_SYSTEM_PROMPT = """You are a helpful, knowledgeable AI assistant. You can discuss any topic — \
programming, data science, machine learning, mathematics, writing, business analysis, \
SAP technologies, general science, history, or anything else the user asks about.

Communication guidelines:
- Give direct, accurate answers to whatever question is asked
- Use code examples when they help clarify a point
- Break complex topics into clear, logical steps
- Be honest when you are uncertain about something
- When including links, always use markdown format: [Descriptive Title](url) — never paste raw URLs as the label

You are not restricted to any particular application or schema. Answer freely and helpfully."""


class GlobalChatAgent:
    """General-purpose AI assistant — answers any question without app-context restrictions."""

    def __init__(self):
        from app.config import get_settings
        settings = get_settings()
        self._request_count = 0

        if getattr(settings, "use_mock_agent", False):
            self._mode = "mock"

        elif settings.llm_provider == "sap_ai_core":
            from app.agents.sap_ai_core_agent import SAPAICoreAuth

            self._mode = "sap"
            self._model_id: str = settings.sap_aicore_model_id or "gpt-4o"
            _is_anthropic = (
                self._model_id.lower().startswith("anthropic--")
                or "claude" in self._model_id.lower()
            )
            _base = settings.sap_aicore_url.rstrip("/")
            _dep  = settings.sap_aicore_deployment_id
            # Always use the orchestration /completion endpoint.
            # Foundation-model deployments are backing resources only.
            self._inference_url = f"{_base}/v2/inference/deployments/{_dep}/completion"
            self._is_anthropic: bool = _is_anthropic
            try:
                self._resource_group: str = settings.sap_aicore_resource_group or "default"
            except Exception:
                self._resource_group = "default"
            self._auth = SAPAICoreAuth(
                auth_url=settings.sap_aicore_auth_url,
                client_id=settings.sap_aicore_client_id,
                client_secret=settings.sap_aicore_client_secret,
            )

        else:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.output_parsers import StrOutputParser

            self._mode = "openai"
            self._model_id = getattr(settings, "openai_model", "gpt-3.5-turbo")
            llm = ChatOpenAI(
                model=self._model_id,
                temperature=0.7,
                api_key=settings.openai_api_key,
                streaming=True,
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", GLOBAL_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
            self._chain = prompt | llm | StrOutputParser()

        logger.info("GlobalChatAgent initialized (mode=%s)", self._mode)

    # ── History helpers ──────────────────────────────────────────────────────

    def _format_history_lc(self, history: Optional[List[Dict]]):
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        out = []
        for msg in (history or []):
            role, content = msg.get("role", "user"), msg.get("content", "")
            if role == "user":
                out.append(HumanMessage(content=content))
            elif role == "assistant":
                out.append(AIMessage(content=content))
            elif role == "system":
                out.append(SystemMessage(content=content))
        return out

    # ── SAP AI Core call ─────────────────────────────────────────────────────

    async def _sap_call(self, message: str, history: Optional[List[Dict]]) -> Dict[str, Any]:
        import aiohttp
        import json as _json

        token = await self._auth.get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "AI-Resource-Group": self._resource_group,
        }

        # Orchestration service format — same for all models.
        template_messages = [{"role": "system", "content": GLOBAL_SYSTEM_PROMPT}]
        for msg in (history or [])[-10:]:
            template_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        template_messages.append({"role": "user", "content": "{{?user_query}}"})
        payload = {
            "orchestration_config": {
                "module_configurations": {
                    "llm_module_config": {
                        "model_name": self._model_id,
                        "model_params": {"max_tokens": 2048, "temperature": 0.7},
                    },
                    "templating_module_config": {"template": template_messages},
                }
            },
            "input_params": {"user_query": message},
        }

        start = datetime.utcnow()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._inference_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise Exception(f"SAP AI Core error {resp.status}: {text}")
                result = _json.loads(text)

        # Orchestration service response parsing
        content = ""
        mr = result.get("module_results", {}).get("llm", {})
        if "choices" in mr:
            content = mr["choices"][0].get("message", {}).get("content", "")
        elif "orchestration_result" in result:
            choices = result["orchestration_result"].get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
        elif "choices" in result:
            content = result["choices"][0].get("message", {}).get("content", "")
        else:
            content = (
                result.get("completion")
                or result.get("text")
                or result.get("output")
                or ""
            )

        return {
            "response": content,
            "model": self._model_id,
            "response_time": (datetime.utcnow() - start).total_seconds(),
        }

    # ── Public interface ─────────────────────────────────────────────────────

    async def get_response(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        self._request_count += 1

        if self._mode == "mock":
            return {"response": f"[Mock] {message}", "model": "mock", "response_time": 0.0}

        if self._mode == "sap":
            return await self._sap_call(message, history)

        # OpenAI
        start = time.time()
        response = await self._chain.ainvoke({
            "input": message,
            "history": self._format_history_lc(history),
        })
        return {
            "response": response,
            "model": self._model_id,
            "response_time": time.time() - start,
        }

    async def stream_response(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        **_kwargs,
    ):
        self._request_count += 1

        if self._mode == "mock":
            yield f"[Mock] {message}"
            return

        if self._mode == "sap":
            result = await self._sap_call(message, history)
            words = result["response"].split(" ")
            for i, word in enumerate(words):
                yield word if i == 0 else " " + word
                await asyncio.sleep(0)
            return

        # OpenAI streaming
        async for chunk in self._chain.astream({
            "input": message,
            "history": self._format_history_lc(history),
        }):
            text = chunk if isinstance(chunk, str) else getattr(chunk, "content", "") or ""
            if text:
                yield text

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_type": "global",
            "status": "healthy",
            "model": getattr(self, "_model_id", "unknown"),
            "mode": self._mode,
            "total_requests": self._request_count,
        }
