from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from app.models.chat import ChatRequest, ChatResponse, AgentStatus
from app.agents.chat_agent import ChatAgent
from app.agents.mock_agent import MockChatAgent
from app.agents.sap_ai_core_agent import SAPAICoreAgent
from app.auth.security import get_current_user
from app.config import get_settings
from app.utils.file_parser import extract_text, validate_file
import asyncio
import json
import time
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Initialize the agent (singleton)
try:
    if settings.use_mock_agent:
        logger.warning("MOCK MODE - Using MockChatAgent for testing")
        chat_agent = MockChatAgent()
    elif settings.llm_provider == "sap_ai_core":
        logger.info("SAP AI Core mode enabled")
        if not all([settings.sap_aicore_url, settings.sap_aicore_client_id, settings.sap_aicore_client_secret]):
            raise ValueError("SAP AI Core requires: SAP_AICORE_URL, SAP_AICORE_CLIENT_ID, SAP_AICORE_CLIENT_SECRET")
        chat_agent = SAPAICoreAgent(
            url=settings.sap_aicore_url,
            client_id=settings.sap_aicore_client_id,
            client_secret=settings.sap_aicore_client_secret,
            model_id=settings.sap_aicore_model_id,
            deployment_id=settings.sap_aicore_deployment_id,
            auth_url=settings.sap_aicore_auth_url,
        )
    else:
        chat_agent = ChatAgent()
    logger.info("Chat agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chat agent: {e}")
    chat_agent = None


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest, current_user=Depends(get_current_user)) -> ChatResponse:
    """Send a message and get a response (requires auth)"""
    if chat_agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat service not available")

    try:
        history = None
        if request.conversation_history:
            history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]

        result = await chat_agent.get_response(message=request.message, history=history)

        return ChatResponse(
            response=result["response"],
            model=result.get("model", "gpt-4"),
            response_time=result.get("response_time"),
            tokens_used=None,
            conversation_id=None
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest, current_user=Depends(get_current_user)):
    if chat_agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat service not available")

    async def event_generator():
        try:
            history = None
            if request.conversation_history:
                history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]

            start_time = time.time()

            if hasattr(chat_agent, 'stream_response'):
                async for chunk in chat_agent.stream_response(message=request.message, history=history):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            else:
                result = await chat_agent.get_response(message=request.message, history=history)
                words = result["response"].split(" ")
                for i, word in enumerate(words):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': word if i == 0 else ' ' + word})}\n\n"
                    await asyncio.sleep(0.03)

            response_time = time.time() - start_time
            model_name = getattr(getattr(chat_agent, 'llm', None), 'model_name', None) or getattr(chat_agent, 'model_id', 'unknown')

            yield f"data: {json.dumps({'type': 'done', 'model': model_name, 'response_time': round(response_time, 2)})}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/upload")
async def chat_with_file(
    file: UploadFile = File(...),
    message: str = Form(default=""),
    conversation_history: str = Form(default="[]"),
    current_user=Depends(get_current_user),
):
    """Upload a file, extract its text, and stream a response about it"""
    if chat_agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat service not available")

    content = await file.read()
    valid, error_msg = validate_file(file.filename or "unknown", len(content))
    if not valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

    try:
        file_text = await extract_text(file.filename or "unknown", content)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    max_chars = 15000
    truncated = len(file_text) > max_chars
    if truncated:
        file_text = file_text[:max_chars] + "\n\n... (truncated)"

    user_prompt = message.strip() if message.strip() else "Please analyze and explain this file."
    combined_message = f"The user uploaded a file named **{file.filename}**.\n\n**File content:**\n```\n{file_text}\n```\n\n**User's request:** {user_prompt}"

    try:
        history = json.loads(conversation_history) if conversation_history else []
    except json.JSONDecodeError:
        history = []

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'file_info', 'filename': file.filename, 'size': len(content), 'truncated': truncated})}\n\n"

            start_time = time.time()
            parsed_history = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in history] if history else None

            if hasattr(chat_agent, 'stream_response'):
                async for chunk in chat_agent.stream_response(message=combined_message, history=parsed_history):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            else:
                result = await chat_agent.get_response(message=combined_message, history=parsed_history)
                words = result["response"].split(" ")
                for i, word in enumerate(words):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': word if i == 0 else ' ' + word})}\n\n"
                    await asyncio.sleep(0.03)

            response_time = time.time() - start_time
            model_name = getattr(getattr(chat_agent, 'llm', None), 'model_name', None) or getattr(chat_agent, 'model_id', 'unknown')
            yield f"data: {json.dumps({'type': 'done', 'model': model_name, 'response_time': round(response_time, 2)})}\n\n"
        except Exception as e:
            logger.error(f"File upload streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    is_healthy = chat_agent is not None
    return {"status": "healthy" if is_healthy else "unhealthy", "service": "chat", "agent_initialized": is_healthy}


@router.get("/status", response_model=AgentStatus, status_code=status.HTTP_200_OK)
async def get_agent_status() -> AgentStatus:
    if chat_agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat agent not initialized")
    try:
        return AgentStatus(**chat_agent.get_status())
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))