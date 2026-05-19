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
import base64
import json
import re
import time
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/chat", tags=["chat"])


_DOC_KEYWORD_GATE = re.compile(
    r'\b(document|doc|word|pdf|excel|xlsx|docx|spreadsheet|report|file)\b',
    re.IGNORECASE,
)

_DOC_PATTERN = re.compile(
    r"\b(generate|create|make|write|produce|build|draft|give|provide|show|prepare|output)\b"
    r"(?:\s+(?:me|us|for\s+me|for\s+us|a|an))?"
    r".{0,80}"
    r"\b(word(?:[\s\-]?doc(?:ument)?)?|\.docx"
    r"|pdf(?:[\s\-]?(?:doc(?:ument)?|file|report))?|\.pdf"
    r"|excel(?:[\s\-]?(?:sheet|file|spreadsheet))?|spreadsheet|\.xlsx?)\b",
    re.IGNORECASE,
)

def _detect_doc_intent_regex(message: str) -> str | None:
    """Regex fast-path. Returns 'word'/'pdf'/'excel' or None."""
    m = _DOC_PATTERN.search(message)
    if not m:
        return None
    keyword = m.group(2).lower()
    if any(k in keyword for k in ("word", "docx", "doc")):
        return "word"
    if "pdf" in keyword:
        return "pdf"
    if any(k in keyword for k in ("excel", "spread", "xlsx", "xls")):
        return "excel"
    return None


async def _classify_doc_intent(message: str) -> str | None:
    """
    Detect if the user is requesting a new downloadable document.

    Strategy (cheapest-first):
    1. Keyword gate  – instant reject if no doc-related words
    2. Regex         – instant accept for obvious explicit forms
    3. LLM call      – single ~30-token call for ambiguous phrasings
                       e.g. "can you give me a word document about X"
    """
    # Step 1 – no document keywords at all → definitely not a doc request
    if not _DOC_KEYWORD_GATE.search(message):
        return None

    # Step 2 – regex fast-path (no LLM cost)
    regex_result = _detect_doc_intent_regex(message)
    if regex_result:
        return regex_result

    # Step 3 – LLM classifier for natural phrasings the regex can't catch
    if chat_agent is None:
        return None

    classify_prompt = (
        "You are a single-purpose JSON classifier. "
        "Does this user message ask you to generate/create/write/give/provide a new downloadable document file?\n\n"
        f"Message: {json.dumps(message)}\n\n"
        "Reply with ONLY one JSON object, no other text:\n"
        '{"doc":true,"type":"word"}   <- Word/.docx requested\n'
        '{"doc":true,"type":"pdf"}    <- PDF requested\n'
        '{"doc":true,"type":"excel"}  <- Excel/spreadsheet requested\n'
        '{"doc":false}                <- no document requested\n\n'
        "Rules: doc=true only when the user explicitly wants a NEW file to download. "
        'Default type is "word" when format is not specified. '
        '"give me a word document about X" -> {"doc":true,"type":"word"}. '
        '"will generate the PDF" (internal process reference) -> {"doc":false}.'
    )

    try:
        result = await chat_agent.get_response(
            message=classify_prompt,
            history=[],
            app_id=None,
        )
        text = (result.get("response") or "").strip()
        m = re.search(r'\{[^}]*"doc"\s*:\s*(true|false)[^}]*\}', text)
        if m:
            data = json.loads(m.group())
            if data.get("doc"):
                doc_type = (data.get("type") or "word").lower()
                return doc_type if doc_type in ("word", "pdf", "excel") else "word"
        return None
    except Exception as ex:
        logger.warning(f"LLM doc-intent classification failed: {ex}")
        return None

async def _generate_doc_event(message: str, doc_type: str, app_id: str | None) -> dict:
    """Build and base64-encode the requested document via LLM (used for Excel). Returns a dict for the SSE event."""
    from app.api.documents import _get_document_content, BUILDERS, EXTENSIONS
    data = await _get_document_content(message, doc_type, None)
    file_bytes = BUILDERS[doc_type](data)
    safe_name = re.sub(r"[^\w\-]", "_", (data.get("title") or message)[:50])
    filename = f"{safe_name}.{EXTENSIONS[doc_type]}"
    return {
        "type": "document",
        "doc_type": doc_type,
        "filename": filename,
        "title": data.get("title", safe_name),
        "content_base64": base64.b64encode(file_bytes).decode(),
    }


_LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]*\)')


def _markdown_to_doc_data(text: str) -> dict:
    """Convert a markdown string into the data structure consumed by _build_word / _build_pdf."""
    lines = text.split('\n')
    title = "Document"
    sections: list[dict] = []
    cur: dict | None = None
    bullets: list[str] = []
    tbl_headers: list[str] = []
    tbl_rows: list[list[str]] = []
    in_code = False
    got_title = False

    def _clean(s: str) -> str:
        s = _LINK_RE.sub(r'\1', s)
        s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
        s = re.sub(r'\*(.+?)\*', r'\1', s)
        s = re.sub(r'`([^`]+)`', r'\1', s)
        return s.strip()

    def _flush_table():
        nonlocal tbl_headers, tbl_rows
        if cur and tbl_headers:
            cur['table'] = {'headers': tbl_headers, 'rows': [r for r in tbl_rows]}
        tbl_headers, tbl_rows = [], []

    def _flush_bullets():
        nonlocal bullets
        if cur and bullets:
            cur['bullets'] = list(bullets)
        bullets.clear()

    def _save_cur():
        nonlocal cur
        if cur:
            _flush_bullets()
            _flush_table()
            sections.append(cur)
            cur = None

    for line in lines:
        s = line.strip()
        if s.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            continue
        if s.startswith('# '):
            ht = _clean(s[2:])
            if not got_title:
                title = ht
                got_title = True
            else:
                _save_cur()
                cur = {'heading': ht, 'level': 1, 'content': '', 'bullets': None, 'table': None}
            continue
        if s.startswith('## '):
            _save_cur()
            cur = {'heading': _clean(s[3:]), 'level': 1, 'content': '', 'bullets': None, 'table': None}
            continue
        if s.startswith('### ') or s.startswith('#### '):
            pl = 4 if s.startswith('### ') else 5
            _save_cur()
            cur = {'heading': _clean(s[pl:]), 'level': 2, 'content': '', 'bullets': None, 'table': None}
            continue
        if not s or re.match(r'^-{3,}$', s):
            continue
        if re.match(r'^\|[\s\-:|]+\|$', s):
            continue
        if s.startswith('|') and s.endswith('|'):
            _flush_bullets()
            cells = [_clean(c) for c in s[1:-1].split('|')]
            if not tbl_headers:
                tbl_headers = cells
            else:
                tbl_rows.append(cells)
            continue
        if tbl_headers:
            _flush_table()
        bm = re.match(r'^(?:[-*+]|\d+\.)\s+(.*)', s)
        if bm:
            bullets.append(_clean(bm.group(1)))
            continue
        if re.match(r'^\d+\.\s+\[', s):
            continue
        _flush_bullets()
        if cur:
            cur['content'] = (cur['content'] + ' ' + _clean(s)).strip()

    _save_cur()
    if not sections:
        sections = [{'heading': 'Content', 'level': 1, 'content': text[:3000], 'bullets': None, 'table': None}]
    return {'title': title, 'subtitle': None, 'sections': sections, 'conclusion': None}


def _build_doc_from_response(text: str, doc_type: str, original_message: str) -> dict:
    """Convert the already-streamed LLM markdown response directly to a document (no second LLM call)."""
    from app.api.documents import BUILDERS, EXTENSIONS
    data = _markdown_to_doc_data(text)
    safe_name = re.sub(r'[^\w\-]', '_', (data.get('title') or original_message)[:50])
    filename = f"{safe_name}.{EXTENSIONS[doc_type]}"
    file_bytes = BUILDERS[doc_type](data)
    return {
        'type': 'document',
        'doc_type': doc_type,
        'filename': filename,
        'title': data.get('title', safe_name),
        'content_base64': base64.b64encode(file_bytes).decode(),
    }


def _short_overview_from_data(data: dict, doc_type: str) -> str:
    """Build a rich overview from structured doc data — title, what's covered, section previews."""
    title = data.get("title", "Document")

    if doc_type == "excel":
        sheets = data.get("sheets", [])
        names = [s.get("name", "") for s in sheets if s.get("name")]
        sheet_count = len(sheets)
        total_rows = sum(len(s.get("rows", [])) for s in sheets)

        overview = f"I've created **{title}** for you as an Excel spreadsheet."
        if sheet_count:
            s = "s" if sheet_count > 1 else ""
            overview += (
                f"\n\nThe file contains **{sheet_count} worksheet{s}**"
            )
            if names:
                overview += f": **{names[0]}**"
                for n in names[1:-1]:
                    overview += f", **{n}**"
                if len(names) > 1:
                    overview += f" and **{names[-1]}**"
            overview += "."
        if total_rows:
            overview += f" Across all sheets there are **{total_rows} data rows** ready for analysis, filtering, and charting."
        for sh in sheets[:2]:
            desc = sh.get("description", "").strip()
            if desc:
                overview += f"\n\n- **{sh.get('name', 'Sheet')}**: {desc}"
        overview += "\n\nThe spreadsheet is fully formatted with colour-coded headers and auto-fitted columns. Download it using the button below."

    else:  # word / pdf
        sections = data.get("sections", [])
        headings = [s.get("heading", "") for s in sections if s.get("heading")]
        section_count = len(sections)
        subtitle = data.get("subtitle", "")

        overview = f"I've created **{title}**"
        if subtitle:
            overview += f" — *{subtitle}*"
        overview += " for you."

        if section_count:
            s = "s" if section_count > 1 else ""
            overview += f"\n\nThe document is **{section_count} section{s}** long and covers:"

        # List each section with a one-sentence preview drawn from its content
        for sec in sections[:8]:
            heading = sec.get("heading", "")
            if not heading:
                continue
            preview = ""
            content = (sec.get("content") or "").strip()
            bullets = sec.get("bullets") or []
            tbl = sec.get("table")
            if content:
                # First sentence only
                m = re.match(r'([^.!?]{20,}[.!?])', content)
                preview = m.group(1).strip() if m else content[:120]
            elif bullets:
                preview = f"Covers: {'; '.join(str(b) for b in bullets[:3])}"
                if len(bullets) > 3:
                    preview += f" and {len(bullets) - 3} more"
            elif tbl:
                cols = tbl.get("headers", [])
                if cols:
                    preview = f"Table: {', '.join(str(c) for c in cols[:4])}"
            if preview:
                overview += f"\n- **{heading}**: {preview}"
            else:
                overview += f"\n- **{heading}**"

        if section_count > 8:
            overview += f"\n- *…and {section_count - 8} more section{'s' if section_count - 8 > 1 else ''}*"

        conclusion = (data.get("conclusion") or "").strip()
        if conclusion:
            m = re.match(r'([^.!?]{20,}[.!?])', conclusion)
            if m:
                overview += f"\n\n{m.group(1).strip()}"

        overview += "\n\nThe full document is ready to download using the button below."

    return overview

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

        result = await chat_agent.get_response(message=request.message, history=history, app_id=request.app_id)

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
            doc_type = await _classify_doc_intent(request.message)
            model_name = (
                getattr(getattr(chat_agent, 'llm', None), 'model_name', None)
                or getattr(chat_agent, 'model_id', 'unknown')
            )

            if doc_type:
                # ── Document generation path ──────────────────────────────────────
                # 1. Signal the frontend immediately so it can show the spinner
                yield f"data: {json.dumps({'type': 'doc_generating', 'doc_type': doc_type})}\n\n"

                try:
                    from app.api.documents import _get_document_content, BUILDERS, EXTENSIONS
                    # One LLM call for the full structured content
                    data = await _get_document_content(request.message, doc_type, None)

                    # 2. Stream a short overview (no extra LLM call)
                    overview = _short_overview_from_data(data, doc_type)
                    words = overview.split(" ")
                    for i, word in enumerate(words):
                        yield f"data: {json.dumps({'type': 'chunk', 'content': word if i == 0 else ' ' + word})}\n\n"
                        await asyncio.sleep(0)

                    response_time = time.time() - start_time
                    yield f"data: {json.dumps({'type': 'done', 'model': model_name, 'response_time': round(response_time, 2)})}\n\n"

                    # 3. Build and emit the document (fast — data already in memory)
                    safe_name = re.sub(r'[^\w\-]', '_', (data.get('title') or request.message)[:50])
                    filename = f"{safe_name}.{EXTENSIONS[doc_type]}"
                    file_bytes = BUILDERS[doc_type](data)
                    doc_event = {
                        'type': 'document',
                        'doc_type': doc_type,
                        'filename': filename,
                        'title': data.get('title', safe_name),
                        'content_base64': base64.b64encode(file_bytes).decode(),
                    }
                    yield f"data: {json.dumps(doc_event)}\n\n"

                except Exception as doc_err:
                    logger.error(f"Document generation failed: {doc_err}", exc_info=True)
                    # Still emit done so the frontend completes the stream
                    response_time = time.time() - start_time
                    yield f"data: {json.dumps({'type': 'done', 'model': model_name, 'response_time': round(response_time, 2)})}\n\n"
                    yield f"data: {json.dumps({'type': 'document_error', 'message': str(doc_err)})}\n\n"

            else:
                # ── Normal chat path ──────────────────────────────────────────────
                if hasattr(chat_agent, 'stream_response'):
                    async for chunk in chat_agent.stream_response(message=request.message, history=history, app_id=request.app_id):
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                else:
                    result = await chat_agent.get_response(message=request.message, history=history, app_id=request.app_id)
                    words = result["response"].split(" ")
                    for i, word in enumerate(words):
                        yield f"data: {json.dumps({'type': 'chunk', 'content': word if i == 0 else ' ' + word})}\n\n"
                        await asyncio.sleep(0.03)

                response_time = time.time() - start_time
                yield f"data: {json.dumps({'type': 'done', 'model': model_name, 'response_time': round(response_time, 2)})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
    )


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
                async for chunk in chat_agent.stream_response(message=combined_message, history=parsed_history, app_id=None):
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


@router.post("/generate-title")
async def generate_title(request: ChatRequest, current_user=Depends(get_current_user)):
    """Generate a short conversation title from the first message using AI"""
    if chat_agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat service not available")

    prompt = f"Generate a short, concise title (max 6 words) for a conversation that starts with this message. Return ONLY the title, no quotes, no extra text.\n\nMessage: {request.message}"

    try:
        result = await chat_agent.get_response(message=prompt, history=None)
        title = result.get("response", "").strip().strip('"').strip("'")
        if not title or len(title) > 60:
            title = request.message[:40].rsplit(" ", 1)[0] + "..."
        return {"title": title}
    except Exception as e:
        logger.error(f"Title generation error: {e}")
        title = request.message[:40].rsplit(" ", 1)[0] + "..."
        return {"title": title}


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