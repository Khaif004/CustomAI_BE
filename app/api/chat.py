"""
Chat API endpoints

This module provides REST API endpoints for chatting with the AI agent.

Endpoints:
- POST /api/chat/ - Send a message and get a response
- GET /api/chat/health - Check if chat service is healthy
- GET /api/chat/status - Get agent status and statistics

Learning: This demonstrates how to integrate LangChain agents with FastAPI
"""

from fastapi import APIRouter, HTTPException, status, Depends
from app.models.chat import ChatRequest, ChatResponse, AgentStatus
from app.agents.chat_agent import ChatAgent
from app.agents.mock_agent import MockChatAgent
from app.agents.sap_ai_core_agent import SAPAICoreAgent
from app.auth.security import get_current_user
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

# Create router with prefix and tags for API organization
# Prefix: All endpoints will be under /api/chat/
# Tags: Groups endpoints in Swagger UI documentation
router = APIRouter(
    prefix="/api/chat",
    tags=["chat"]
)

# Initialize the agent (singleton pattern)
# This creates ONE instance that handles all requests
# More efficient than creating a new agent for each request
try:
    if settings.use_mock_agent:
        logger.warning("⚠️  MOCK MODE ENABLED - Using MockChatAgent for testing")
        chat_agent = MockChatAgent()
    elif settings.llm_provider == "sap_ai_core":
        logger.info("🚀 SAP AI CORE MODE - Using SAP AI Core for inference")
        if not all([settings.sap_aicore_url, settings.sap_aicore_client_id, settings.sap_aicore_client_secret]):
            raise ValueError("SAP AI Core requires: SAP_AICORE_URL, SAP_AICORE_CLIENT_ID, SAP_AICORE_CLIENT_SECRET")
        chat_agent = SAPAICoreAgent(
            url=settings.sap_aicore_url,
            client_id=settings.sap_aicore_client_id,
            client_secret=settings.sap_aicore_client_secret,
            model_id=settings.sap_aicore_model_id,
            deployment_id=settings.sap_aicore_deployment_id,
            auth_url=settings.sap_aicore_auth_url,  # Pass auth URL if provided
        )
    else:
        logger.info("Using OpenAI API")
        chat_agent = ChatAgent()
    logger.info("Chat agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chat agent: {str(e)}")
    chat_agent = None


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest, current_user = Depends(get_current_user)) -> ChatResponse:
    """
    Chat with the AI assistant
    
    ⚠️ **REQUIRES AUTHENTICATION** - Pass JWT token in Authorization header
    
    This is the main endpoint for user interaction.
    
    Flow:
    1. Validate JWT token from Authorization header
    2. Receive user message and optional history
    3. Validate request using Pydantic (automatic)
    4. Pass to ChatAgent
    5. Return structured response
    
    Args:
        request: ChatRequest with message and optional conversation history
        current_user: Authenticated user from JWT token (automatic)
        
    Returns:
        ChatResponse with AI's reply and metadata
        
    Raises:
        HTTPException: If authentication fails, agent not initialized, or request fails
        
    Example Request with Auth:
        ```bash
        curl -X POST http://localhost:8000/api/chat/ \\
          -H "Authorization: Bearer YOUR_TOKEN_HERE" \\
          -H "Content-Type: application/json" \\
          -d '{
            "message": "What is SAP BTP?",
            "conversation_history": []
          }'
        ```
        
    Example Response:
        ```json
        {
            "response": "SAP BTP is a cloud platform...",
            "model": "gpt-4",
            "response_time": 1.23,
            "tokens_used": null
        }
        ```
    """
    # Check if agent is initialized
    if chat_agent is None:
        logger.error("Chat agent not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not available. Agent failed to initialize."
        )
    
    try:
        # Log incoming request with user info
        message_preview = request.message[:100] + "..." if len(request.message) > 100 else request.message
        logger.info(f"Chat request from user {current_user.get('username', 'unknown')}: '{message_preview}'")
        
        # Convert conversation history to dict format
        history = None
        if request.conversation_history:
            history = [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in request.conversation_history
            ]
            logger.debug(f"Including {len(history)} messages from history")
        
        # Get response from agent
        result = await chat_agent.get_response(
            message=request.message,
            history=history
        )
        
        # Build response
        response = ChatResponse(
            response=result["response"],
            model=result.get("model", "gpt-4"),
            response_time=result.get("response_time"),
            tokens_used=None,  # TODO: Implement token counting
            conversation_id=None  # TODO: Implement conversation tracking
        )
        
        logger.info(f"✓ Response sent to {current_user.get('username')} (time: {result.get('response_time', 0):.2f}s)")
        return response
        
    except Exception as e:
        # Log error with full traceback
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        
        # Return user-friendly error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}"
        )


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint for chat service
    
    This endpoint is used by:
    - Load balancers to check if service is up
    - Monitoring systems to track availability
    - Deployment pipelines for readiness checks
    
    Returns:
        Dict with service status
        
    Example Response:
        ```json
        {
            "status": "healthy",
            "service": "chat",
            "agent_initialized": true
        }
        ```
    """
    is_healthy = chat_agent is not None
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "chat",
        "agent_initialized": is_healthy
    }


@router.get("/status", response_model=AgentStatus, status_code=status.HTTP_200_OK)
async def get_agent_status() -> AgentStatus:
    """
    Get detailed status and statistics about the chat agent
    
    Useful for:
    - Monitoring agent performance
    - Tracking usage
    - Debugging issues
    
    Returns:
        AgentStatus with detailed information
        
    Raises:
        HTTPException: If agent is not initialized
        
    Example Response:
        ```json
        {
            "agent_type": "chat",
            "status": "healthy",
            "model": "gpt-4",
            "total_requests": 42,
            "last_request_time": "2026-03-18T07:30:00"
        }
        ```
    """
    if chat_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat agent not initialized"
        )
    
    try:
        status_info = chat_agent.get_status()
        return AgentStatus(**status_info)
    except Exception as e:
        logger.error(f"Error getting agent status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent status: {str(e)}"
        )


# Additional endpoint ideas for future implementation:
# 
# @router.post("/stream")
# async def chat_stream(request: ChatRequest):
#     """Stream responses token by token for better UX"""
#     pass
#
# @router.get("/conversations/{conversation_id}")
# async def get_conversation(conversation_id: str):
#     """Retrieve a past conversation"""
#     pass
#
# @router.delete("/conversations/{conversation_id}")
# async def delete_conversation(conversation_id: str):
#     """Delete a conversation"""
#     pass