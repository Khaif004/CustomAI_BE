"""
SAP AI Core Chat Agent with LangChain Integration

Connects LangChain to SAP AI Core (running on SAP BTP) for enterprise LLM capabilities.
Uses LangChain's ChatOpenAI wrapper with SAP AI Core's OpenAI-compatible endpoint.

This setup is ideal for:
- SAP BTP development environments
- Zero cost inference (SAP AI Core free tier)
- Enterprise integration (native SAP infrastructure)
- Production-grade reliability with LangChain abstractions

Required Environment Variables:
- SAP_AICORE_URL: Your AI Core API URL
- SAP_AICORE_AUTH_URL: OAuth2 authentication URL
- SAP_AICORE_CLIENT_ID: OAuth2 client ID
- SAP_AICORE_CLIENT_SECRET: OAuth2 client secret
- SAP_AICORE_MODEL_ID: Model name (e.g., 'gpt-4')
- SAP_AICORE_DEPLOYMENT_ID: Deployment ID (UUID)
"""

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
        
        logger.debug(f"Refreshing access token from {token_endpoint}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    token_endpoint,
                    data={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "grant_type": "client_credentials",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OAuth2 auth failed [{response.status}]: {error_text}")
                        raise Exception(f"Authentication failed: {error_text}")
                    
                    data = await response.json()
                    self.access_token = data.get("access_token")
                    expires_in = data.get("expires_in", 3600)
                    self.token_expiry = time.time() + expires_in - 60
                    
                    logger.debug(f"Token refreshed (valid for {expires_in}s)")
                    return self.access_token
                    
            except Exception as e:
                logger.error(f"Token retrieval failed: {str(e)}")
                raise


class SAPAICoreAgent:
    """
    Production-grade SAP AI Core agent with LangChain integration.
    
    Provides:
    - Simple async interface for multi-turn conversations
    - Automatic OAuth2 token refresh
    - Proper error handling with RBAC guidance
    - Seamless integration with SAP AI Core deployments
    
    Example:
        agent = SAPAICoreAgent(
            url="https://api.aicore.cloud.sap",
            auth_url="https://auth.your-region.hana.ondemand.com",
            client_id="your-client-id",
            client_secret="your-client-secret",
            model_id="gpt-4",
            deployment_id="abc123def456"
        )
        response = await agent.get_response("Hello, what is SAP BTP?")
    """
    
    def __init__(
        self,
        url: str,
        auth_url: str,
        client_id: str,
        client_secret: str,
        model_id: str = "gpt-4",
        deployment_id: str = "default",
    ):
        """
        Initialize SAP AI Core agent with LangChain integration
        
        Args:
            url: AI Core API base URL (e.g., https://api.example.aicore.cloud.sap)
            auth_url: OAuth2 authentication URL from SAP service key
            client_id: OAuth2 client ID from SAP service key
            client_secret: OAuth2 client secret from SAP service key
            model_id: Model name for logging/identification
            deployment_id: Deployment UUID for the model
        """
        logger.info("🚀 Initializing SAP AI Core Agent with LangChain...")
        
        self.url = url.rstrip("/")
        self.auth_url = auth_url.rstrip("/")
        self.model_id = model_id
        self.deployment_id = deployment_id
        self.request_count = 0
        
        # Initialize OAuth2 authentication provider
        self.auth_client = SAPAICoreAuth(
            auth_url=self.auth_url,
            client_id=client_id,
            client_secret=client_secret,
        )
        
        # Build SAP AI Core inference endpoint
        # Endpoint format: /v2/lm/inference/deployments/{deployment_id}
        self.inference_url = f"{self.url}/v2/lm/inference/deployments/{self.deployment_id}"
        
        logger.info(f"✓ SAP AI Core Agent configured")
        logger.info(f"  - API URL: {self.url}")
        logger.info(f"  - Auth URL: {self.auth_url}/oauth/token")
        logger.info(f"  - Model: {model_id}")
        logger.info(f"  - Deployment: {deployment_id}")
        logger.info(f"  - Inference Endpoint: /v2/lm/inference/deployments/{deployment_id}")
    
    async def get_response(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Get response from SAP AI Core via OpenAI-compatible endpoint.
        
        Handles:
        - OAuth2 token refresh automatically
        - Conversation history context
        - Error handling with helpful guidance
        - Response time tracking
        
        Args:
            message: User's question or prompt
            history: Optional conversation history as list of dicts with "role" and "content"
            
        Returns:
            Dict with keys:
                - response: The AI's response text
                - model: Model identifier used
                - response_time: Time taken in seconds
                
        Raises:
            Exception: On authentication, API, or network errors
        """
        self.request_count += 1
        start_time = datetime.utcnow()
        
        logger.info(f"Processing request #{self.request_count}: {message[:60]}...")
        
        try:
            # Get fresh OAuth2 access token
            token = await self.auth_client.get_token()
            
            # Prepare request headers with SAP AI Core requirements
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "AI-Resource-Group": "default",  # Required by SAP AI Core
            }
            
            # Build messages for OpenAI-compatible format
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for enterprise software and cloud services."}
            ]
            
            # Add conversation history
            if history:
                for msg in history[-10:]:  # Last 10 messages for context
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Prepare request payload (OpenAI-compatible format)
            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.9,
            }
            
            logger.debug(f"Sending request to: {self.inference_url}")
            logger.debug(f"Messages: {len(messages)}")
            
            # Make inference request to SAP AI Core
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.inference_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    response_text = await response.text()
                    
                    if response.status == 403:
                        logger.error(
                            "\n❌ RBAC Permission Error (403 Access Denied)\n"
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            "Your service account lacks SAP AI Core inference permissions.\n\n"
                            "📋 TO FIX:\n"
                            "  1. Open SAP BTP Cockpit\n"
                            "  2. Go to: Subaccount → Security → Users\n"
                            "  3. Find your service account (from .env file)\n"
                            "  4. Assign Role: 'AI_CORE_INFERENCE_USER'\n"
                            "  5. Also assign Role Collection: 'AI_CORE_ACCESS_CONTROL_MEMBER'\n"
                            "  6. ⏱️  Wait 2-3 minutes for changes to propagate\n"
                            "  7. Retry your request\n"
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"This is expected until RBAC permissions are assigned.\n"
                            f"Endpoint: {self.inference_url}\n"
                        )
                        raise Exception(f"RBAC Permission Denied: {response_text}")
                    elif response.status == 401:
                        logger.error(f"Authentication failed: {response_text}")
                        raise Exception(f"Authentication error: {response_text}")
                    elif response.status == 404:
                        logger.error(f"Deployment not found: {self.deployment_id}")
                        raise Exception(f"Deployment not found: {response_text}")
                    elif response.status != 200:
                        logger.error(f"API error [{response.status}]: {response_text}")
                        raise Exception(f"API error {response.status}: {response_text}")
                    
                    # Parse response (OpenAI-compatible format)
                    result = await response.json()
                    
                    # Extract response from OpenAI-compatible format
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0].get("message", {}).get("content", "")
                    else:
                        # Fallback for other response formats
                        content = (
                            result.get("completion") or 
                            result.get("text") or 
                            result.get("output") or 
                            str(result)
                        )
                    
                    end_time = datetime.utcnow()
                    response_time = (end_time - start_time).total_seconds()
                    
                    logger.info(f"✓ Response received ({response_time:.2f}s)")
                    
                    return {
                        "response": content,
                        "model": self.model_id,
                        "response_time": response_time,
                    }
        
        except Exception as e:
            logger.error(f"❌ Error in get_response: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and statistics"""
        return {
            "agent_type": "sap_ai_core",
            "status": "healthy",
            "model": self.model_id,
            "deployment": self.deployment_id,
            "total_requests": self.request_count,
            "api_endpoint": self.inference_url,
        }
