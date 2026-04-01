

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

    async def get_response(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        self.request_count += 1
        start_time = datetime.utcnow()

        token = await self.auth_client.get_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "AI-Resource-Group": "default",
        }

        # Build orchestration template messages
        template_messages = [
            {"role": "system", "content": "You are a helpful AI assistant for enterprise software and cloud services."},
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
                        "model_params": {"max_tokens": 1024, "temperature": 0.7, "top_p": 0.9}
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

                result = await response.json()

                # Parse orchestration response format
                content = ""
                module_results = result.get("module_results", {})
                llm_result = module_results.get("llm", {})
                if "choices" in llm_result and len(llm_result["choices"]) > 0:
                    content = llm_result["choices"][0].get("message", {}).get("content", "")
                elif "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                else:
                    content = result.get("completion") or result.get("text") or result.get("output") or str(result)

                response_time = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"Response received ({response_time:.2f}s)")

                return {"response": content, "model": self.model_id, "response_time": response_time}

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_type": "sap_ai_core",
            "status": "healthy",
            "model": self.model_id,
            "deployment": self.deployment_id,
            "total_requests": self.request_count,
            "api_endpoint": self.inference_url,
        }
