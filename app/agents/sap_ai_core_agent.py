

import logging
import asyncio
import os
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

import aiohttp
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)


class SAPAICoreAuth:
    """OAuth2 authentication client for SAP AI Core with token caching"""

    def __init__(self, auth_url: str, client_id: str, client_secret: str):
        self.auth_url = auth_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expiry = 0

    async def get_token(self) -> str:
        """Get valid OAuth2 access token (cached and auto-refreshed)"""
        if self.access_token and time.time() < self.token_expiry:
            return self.access_token

        token_endpoint = (
            self.auth_url
            if self.auth_url.endswith("/oauth/token")
            else f"{self.auth_url}/oauth/token"
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                token_endpoint,
                data={"client_id": self.client_id, "client_secret": self.client_secret, "grant_type": "client_credentials"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Authentication failed: {error_text}")

                data = await response.json()
                self.access_token = data.get("access_token")
                expires_in = data.get("expires_in", 3600)
                self.token_expiry = time.time() + expires_in - 60
                return self.access_token


class SAPAICoreAgent:
    """SAP AI Core agent with orchestration service integration"""

    def __init__(self, url: str, auth_url: str, client_id: str, client_secret: str,
                 model_id: str = "gpt-4o", deployment_id: str = "default"):
        self.url = url.rstrip("/")
        self.model_id = model_id
        self.deployment_id = deployment_id
        self.request_count = 0

        self.auth_client = SAPAICoreAuth(auth_url=auth_url, client_id=client_id, client_secret=client_secret)
        self.inference_url = f"{self.url}/v2/inference/deployments/{self.deployment_id}/completion"

        logger.info(f"SAP AI Core Agent configured - model: {model_id}, deployment: {deployment_id}")

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

    def _build_system_message(self, rag_context: Optional[str] = None, app_id: Optional[str] = None) -> str:
        base = (
            "You are a helpful AI assistant for enterprise software and cloud services. "
            "When including links, always use markdown format with a descriptive human-readable title "
            "as the link text — never use the raw URL as the label. "
            "Example: [SAP BTP Documentation](https://help.sap.com/docs/btp) "
            "not [https://help.sap.com/docs/btp](https://help.sap.com/docs/btp)."
        )
        if rag_context:
            base += (
                f"\n\nYou are currently assisting a user inside the '{app_id}' application. "
                "Use the following retrieved context from that application to answer accurately. "
                "If the question is general, answer normally. If it's about this application's data "
                "or entities, prioritise the context below.\n\n"
                + rag_context
            )
        return base

    async def get_response(self, message: str, history: Optional[List[Dict[str, str]]] = None, app_id: Optional[str] = None) -> Dict[str, Any]:
        self.request_count += 1
        start_time = datetime.utcnow()

        rag_context = await self._fetch_rag_context(message, app_id)
        system_message = self._build_system_message(rag_context, app_id)

        token = await self.auth_client.get_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "AI-Resource-Group": "default",
        }

        # Build orchestration template messages
        template_messages = [
            {"role": "system", "content": system_message},
        ]
        if history:
            for msg in history[-10:]:
                template_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        template_messages.append({"role": "user", "content": "{{?user_query}}"})

        payload = {
            "orchestration_config": {
                "module_configurations": {
                    "llm_module_config": {
                        "model_name": self.model_id,
                        "model_params": {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.9}
                    },
                    "templating_module_config": {"template": template_messages}
                }
            },
            "input_params": {"user_query": message}
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.inference_url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                response_text = await response.text()

                if response.status != 200:
                    raise Exception(f"API error {response.status}: {response_text}")

                import json as _json
                try:
                    result = _json.loads(response_text)
                except Exception:
                    raise Exception(f"Failed to parse API response: {response_text[:500]}")

                # Parse orchestration response — log structure to diagnose extraction path
                logger.info(f"API response keys: {list(result.keys())}")

                content = ""
                # Path 1: module_results.llm.choices (orchestration v1)
                module_results = result.get("module_results", {})
                llm_result = module_results.get("llm", {})
                if "choices" in llm_result and len(llm_result["choices"]) > 0:
                    content = llm_result["choices"][0].get("message", {}).get("content", "")
                    logger.info(f"Extracted via module_results.llm.choices ({len(content)} chars)")
                # Path 2: orchestration_result.choices (orchestration v2)
                elif "orchestration_result" in result:
                    orch = result["orchestration_result"]
                    choices = orch.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        logger.info(f"Extracted via orchestration_result.choices ({len(content)} chars)")
                # Path 3: top-level choices
                elif "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    logger.info(f"Extracted via top-level choices ({len(content)} chars)")
                # Path 4: fallback scalar fields
                else:
                    content = result.get("completion") or result.get("text") or result.get("output") or ""
                    logger.warning(f"Fell back to scalar extraction, result keys: {list(result.keys())}")
                    if not content:
                        # Last resort: dump so we can see the structure
                        logger.error(f"Could not extract content. Full response: {response_text[:1000]}")

                response_time = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"Response received ({response_time:.2f}s)")

                return {"response": content, "model": self.model_id, "response_time": response_time}

    async def stream_response(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        app_id: Optional[str] = None,
    ):
        """Stream response token-by-token using word-level chunking."""
        result = await self.get_response(message=message, history=history, app_id=app_id)
        text = result.get("response", "")
        words = text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            if chunk:
                yield chunk
            await asyncio.sleep(0)

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_type": "sap_ai_core",
            "status": "healthy",
            "model": self.model_id,
            "deployment": self.deployment_id,
            "total_requests": self.request_count,
            "api_endpoint": self.inference_url,
        }
