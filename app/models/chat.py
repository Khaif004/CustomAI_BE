from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    DEVELOPER_HELPER = "developer_helper"
    DATA_ANALYST = "data_analyst"
    ARCHITECT = "architect"
    DOCUMENTATION = "documentation"


class ConversationStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="The message content")
    timestamp: Optional[datetime] = None
    agent_type: Optional[AgentType] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500000)
    conversation_history: Optional[List[ChatMessage]] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = Field(
        None,
        description="Persistent chat session ID. If omitted the backend generates one and returns it in the 'done' SSE event.",
    )
    project_context: Optional[str] = None
    stream: bool = False
    app_id: Optional[str] = Field(
        None,
        description="ID of the host application. When provided, relevant context is retrieved "
                    "from that app's registered knowledge and injected into the LLM prompt.",
        pattern=r"^[a-zA-Z0-9_-]*$",
    )
    fiori_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Context payload sent from an embedded Fiori/UI5 application via postMessage. "
                    "Contains the current entity data, URL hash, appId, and serviceUrl.",
    )
    odata_token: Optional[str] = Field(
        None,
        description="Bearer token forwarded from the host Fiori app (XSUAA or JWT). "
                    "Used by the backend to proxy OData calls on behalf of the user so the "
                    "chatbot can fetch real record counts and data.",
    )


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    model: str = "gpt-4"
    agent_type: Optional[str] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentStatus(BaseModel):
    agent_type: str
    status: str
    model: str
    last_request_time: Optional[datetime] = None
    total_requests: int = 0


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:12]}")
    title: Optional[str] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    messages: List[ChatMessage] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)