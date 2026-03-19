import logging
import json
import asyncio
from typing import Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status, Query
from app.agents.orchestrator import get_orchestrator
from app.models.chat import ChatMessage
from app.knowledge.document_store import DocumentStore
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["websocket"]
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.user_conversations = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected: {user_id} (total: {len(self.active_connections)})")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        if user_id in self.user_conversations:
            del self.user_conversations[user_id]
        logger.info(f"WebSocket disconnected: {user_id} (total: {len(self.active_connections)})")
    
    async def broadcast(self, message: dict, exclude_user: str = None):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error broadcasting message: {str(e)}")
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to specific connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")


manager = ConnectionManager()

orchestrator = get_orchestrator()
doc_store = DocumentStore()


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    user_id: str = Query(...),
    conversation_id: str = Query(None),
    project_context: str = Query(None)
):
    """
    WebSocket endpoint for real-time chat
    
    Query parameters:
        - user_id: User identifier (required)
        - conversation_id: Existing conversation (optional)
        - project_context: Project context (optional)
    
    Message format:
    {
        "type": "message",
        "content": "User message",
        "metadata": {}
    }
    """
    await manager.connect(websocket, user_id)
    
    try:
        # Initialize or get conversation
        if conversation_id:
            conversation = doc_store.get_conversation(conversation_id)
            if not conversation:
                await websocket.send_json({
                    "type": "error",
                    "message": "Conversation not found"
                })
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
        else:
            conversation = doc_store.create_conversation(
                user_id=user_id,
                project_id=project_context
            )
        
        conv_id = conversation.id
        manager.user_conversations[user_id] = conv_id
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "conversation_id": conv_id,
            "message": f"Connected. Conversation ID: {conv_id}"
        })
        
        logger.info(f"User {user_id} connected to conversation {conv_id}")
        
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message_type = message_data.get("type", "message")
            
            if message_type == "message":
                await handle_chat_message(
                    websocket,
                    user_id,
                    conv_id,
                    message_data,
                    project_context
                )
            
            elif message_type == "ping":
                # Respond to ping
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            elif message_type == "close":
                await websocket.send_json({
                    "type": "closing",
                    "message": "Closing connection"
                })
                break
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"User {user_id} disconnected normally")
    
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")
        manager.disconnect(websocket, user_id)


async def handle_chat_message(
    websocket: WebSocket,
    user_id: str,
    conversation_id: str,
    message_data: dict,
    project_context: str = None
):
    """Handle incoming chat message"""
    try:
        query = message_data.get("content", "").strip()
        if not query:
            await websocket.send_json({
                "type": "error",
                "message": "Empty message"
            })
            return
        
        # Send acknowledgment
        await websocket.send_json({
            "type": "ack",
            "message": "Processing your query..."
        })
        
        # Log that we're processing
        logger.debug(f"Processing message for user {user_id}: {query[:100]}")
        
        # Process through orchestrator
        result = await orchestrator.process_query(
            query=query,
            conversation_id=conversation_id,
            user_id=user_id,
            project_context=project_context
        )
        
        if result.get("status") != "success":
            await websocket.send_json({
                "type": "error",
                "message": result.get("message", "Error processing query"),
                "conversation_id": conversation_id
            })
            return
        
        # Send streaming response (can be done in chunks)
        response_message = {
            "type": "response",
            "content": result.get("response", ""),
            "conversation_id": conversation_id,
            "agent_type": result.get("agent", "orchestrator"),
            "agents_used": result.get("agents_used", []),
            "processing_time": result.get("processing_time", 0),
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": result.get("metadata", {})
        }
        
        await websocket.send_json(response_message)
        
        # Optionally broadcast to other users in same project (if project_context provided)
        if project_context:
            broadcast_message = {
                "type": "broadcast",
                "message": f"User {user_id} asked: {query[:100]}...",
                "project": project_context,
                "timestamp": datetime.utcnow().isoformat()
            }
            # Can broadcast to other project members here
        
        logger.debug(f"Successfully processed message for user {user_id}")
    
    except Exception as e:
        logger.error(f"Error handling chat message: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": f"Error processing message: {str(e)}",
            "conversation_id": conversation_id
        })


# ==================== System WebSocket ====================

@router.websocket("/ws/system")
async def websocket_system_endpoint(websocket: WebSocket):
    """
    WebSocket for system status updates and monitoring
    """
    await websocket.accept()
    
    try:
        logger.info("System WebSocket connected")
        
        # Send initial status
        status_info = await orchestrator.get_system_status()
        await websocket.send_json({
            "type": "status_initial",
            "data": status_info
        })
        
        # Send periodic updates
        while True:
            await asyncio.sleep(30)  # Send update every 30 seconds
            
            status_info = await orchestrator.get_system_status()
            await websocket.send_json({
                "type": "status_update",
                "data": status_info,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    except WebSocketDisconnect:
        logger.info("System WebSocket disconnected")
    
    except Exception as e:
        logger.error(f"System WebSocket error: {str(e)}")
