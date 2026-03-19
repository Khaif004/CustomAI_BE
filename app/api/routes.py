"""
Chat API Endpoints

Complete REST API for the multi-agent AI system.
Includes chat, conversation management, health, and system status endpoints.
"""

import logging
import time
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException, status, Query, Depends, BackgroundTasks
from app.models.chat import (
    ChatRequest, ChatResponse, AgentStatus, SystemStatus,
    Conversation, ConversationStatus, ChatMessage,
    BatchChatRequest, BatchChatResponse, HealthStatus
)
from app.agents.orchestrator import get_orchestrator
from app.knowledge.knowledge_base import get_knowledge_base
from app.knowledge.document_store import DocumentStore
from app.config import get_settings
import asyncio

logger = logging.getLogger(__name__)
settings = get_settings()

# Create router
router = APIRouter(
    prefix="/api/chat",
    tags=["chat"]
)

# Initialize
orchestrator = get_orchestrator()
kb = get_knowledge_base()
doc_store = DocumentStore()


# ==================== Chat Endpoints ====================

@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint - Send a message to AI agents
    
    Routes query through supervisor agent to appropriate specialized agents.
    Maintains conversation history for context.
    
    Args:
        request: ChatRequest with message and context
        
    Returns:
        ChatResponse with agent response and metadata
    """
    try:
        start_time = time.time()
        
        logger.info(f"Chat request: {request.message[:100]}")
        
        # Process query through orchestrator
        result = await orchestrator.process_query(
            query=request.message,
            conversation_id=request.conversation_id,
            project_context=request.project_context
        )
        
        if result.get("status") != "success":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to process query")
            )
        
        # Build response
        response = ChatResponse(
            response=result.get("response", ""),
            conversation_id=result.get("conversation_id", ""),
            agent_type=result.get("agent", "supervisor"),
            tokens_used=None,
            response_time=time.time() - start_time,
            sources=result.get("sources", []),
            metadata={
                "agents_used": result.get("agents_used", []),
                "routing_decision": result.get("supervision_routing", {})
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {str(e)}"
        )


@router.post("/batch", response_model=BatchChatResponse)
async def batch_chat(request: BatchChatRequest) -> BatchChatResponse:
    """
    Batch chat endpoint - Process multiple queries
    
    Args:
        request: Batch of chat requests
        
    Returns:
        Batch response with all results
    """
    try:
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        
        logger.info(f"Batch request with {len(request.requests)} queries")
        
        responses = []
        
        if request.process_sequentially:
            # Process sequentially
            for chat_req in request.requests:
                result = await chat_endpoint(chat_req)
                responses.append(result)
        else:
            # Process in parallel
            tasks = [chat_endpoint(req) for req in request.requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            responses = [
                r if not isinstance(r, Exception) else ChatResponse(
                    response=f"Error: {str(r)}",
                    conversation_id="",
                    agent_type="error"
                )
                for r in responses
            ]
        
        batch_response = BatchChatResponse(
            responses=responses,
            batch_id=batch_id,
            processing_time=time.time() - start_time
        )
        
        return batch_response
        
    except Exception as e:
        logger.error(f"Batch chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing error: {str(e)}"
        )


# ==================== Conversation Management ====================

@router.get("/conversations", response_model=List[Conversation])
async def get_conversations(
    user_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> List[Conversation]:
    """Get conversations with filtering and pagination"""
    try:
        status_enum = None
        if status_filter:
            status_enum = ConversationStatus(status_filter)
        
        conversations = doc_store.get_conversations(
            user_id=user_id,
            project_id=project_id,
            status=status_enum,
            limit=limit,
            offset=offset
        )
        
        return conversations
        
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str) -> Conversation:
    """Get specific conversation"""
    try:
        conversation = doc_store.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/conversations", response_model=Conversation)
async def create_conversation(
    title: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None)
) -> Conversation:
    """Create new conversation"""
    try:
        conversation = doc_store.create_conversation(
            title=title,
            project_id=project_id,
            user_id=user_id
        )
        
        return conversation
        
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.put("/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation_status(
    conversation_id: str,
    status: ConversationStatus
) -> Conversation:
    """Update conversation status"""
    try:
        success = doc_store.update_conversation_status(conversation_id, status)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        conversation = doc_store.get_conversation(conversation_id)
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str):
    """Delete conversation"""
    try:
        success = doc_store.delete_conversation(conversation_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== Health & Status ====================

@router.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "chat_api",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@router.get("/status", response_model=dict)
async def system_status():
    """Get system status and statistics"""
    try:
        status_info = await orchestrator.get_system_status()
        
        # Add knowledge base stats
        kb_stats = kb.get_knowledge_stats()
        status_info["knowledge_base"] = kb_stats
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== Knowledge Base ====================

@router.post("/knowledge/search")
async def search_knowledge(
    query: str,
    k: int = Query(5, ge=1, le=20),
    category: Optional[str] = None,
    score_threshold: float = Query(0.0, ge=0.0, le=1.0)
):
    """Search knowledge base"""
    try:
        results = kb.search(
            query=query,
            k=k,
            score_threshold=score_threshold,
            category_filter=category
        )
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Knowledge search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/knowledge/stats")
async def knowledge_stats():
    """Get knowledge base statistics"""
    try:
        return kb.get_knowledge_stats()
    except Exception as e:
        logger.error(f"Error getting KB stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== Documentation ====================

@router.get("/docs/agents")
async def get_agents_documentation():
    """Get documentation about available agents"""
    try:
        from app.agents.supervisor_agent import SupervisorAgent
        supervisor = SupervisorAgent()
        return supervisor.get_available_agents()
    except Exception as e:
        logger.error(f"Error getting agents doc: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/docs/tools")
async def get_tools_documentation():
    """Get documentation about available tools"""
    try:
        from app.tools.registry import get_tools_registry
        registry = get_tools_registry()
        return registry.get_tools_info()
    except Exception as e:
        logger.error(f"Error getting tools doc: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
