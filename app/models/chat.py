"""
Chat models for request/response validation
Pydantic models ensure type safety and automatic API documentation
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ChatMessage(BaseModel):
    """
    Represents a single message in a conversation
    
    Used for conversation history in requests
    """
    role: str = Field(
        ..., 
        description="Message role: 'user', 'assistant', or 'system'",
        examples=["user", "assistant"]
    )
    content: str = Field(
        ..., 
        description="The actual message content",
        examples=["Explain the CAP model"]
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="When this message was created"
    )


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint
    
    Validates incoming chat requests from users
    """
    message: str = Field(
        ..., 
        description="User's input message",
        min_length=1,
        max_length=5000,
        examples=["What is SAP BTP?"]
    )
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous messages in the conversation for context",
        examples=[[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you?"}
        ]]
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response (for future implementation)"
    )
    
    class Config:
        """Pydantic config"""
        json_schema_extra = {
            "example": {
                "message": "Explain the CAP model in SAP BTP",
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "What is SAP BTP?"
                    },
                    {
                        "role": "assistant",
                        "content": "SAP BTP is a cloud platform..."
                    }
                ],
                "stream": False
            }
        }


class ChatResponse(BaseModel):
    """
    Response model for chat endpoint
    
    Ensures consistent response structure
    """
    response: str = Field(
        ..., 
        description="AI assistant's response",
        examples=["The CAP model is a programming model..."]
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this conversation (for future use)"
    )
    model: str = Field(
        default="gpt-4",
        description="The LLM model used to generate the response"
    )
    tokens_used: Optional[int] = Field(
        default=None,
        description="Total tokens used in this request (for cost tracking)"
    )
    response_time: Optional[float] = Field(
        default=None,
        description="Time taken to generate response in seconds"
    )
    
    class Config:
        """Pydantic config"""
        json_schema_extra = {
            "example": {
                "response": "The CAP (Cloud Application Programming) model...",
                "conversation_id": "conv_123456",
                "model": "gpt-4",
                "tokens_used": 150,
                "response_time": 1.23
            }
        }


class AgentStatus(BaseModel):
    """
    Status information about an agent
    
    Used for health checks and monitoring
    """
    agent_type: str = Field(..., description="Type of agent (e.g., 'chat', 'code_analyzer')")
    status: str = Field(..., description="Current status: 'healthy', 'degraded', 'unhealthy'")
    model: str = Field(..., description="LLM model being used")
    last_request_time: Optional[datetime] = Field(default=None, description="Last request timestamp")
    total_requests: int = Field(default=0, description="Total number of requests processed")