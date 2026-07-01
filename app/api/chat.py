from fastapi import APIRouter, HTTPException, Request, status, Depends, UploadFile, File, Form
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
import uuid as _uuid_module
from typing import Optional

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


def _looks_like_shared_document(message: str) -> bool:
    """
    Return True when the message is almost certainly content being SHARED by the
    user (a spec, report, email, article, etc.) rather than a short command.

    Heuristics:
    - Very long messages (> 400 chars) that contain multiple newlines are almost
      always pasted documents, not generation commands.
    - Structural markers such as headings, numbered sections, or table-like lines
      reinforce this further.
    """
    if len(message) < 400:
        return False

    lines = [l for l in message.splitlines() if l.strip()]
    if len(lines) < 5:
        return False

    heading_re = re.compile(r'^#{1,4}\s+\S|^\d+\.\s+[A-Z]|^[A-Z][A-Za-z ]{3,}:?\s*$')
    structural_lines = sum(1 for l in lines if heading_re.match(l.strip()))
    if structural_lines >= 2:
        return True

    if len(message) > 800 and len(lines) >= 8:
        return True

    return False


async def _classify_doc_intent(message: str) -> str | None:
    """
    Detect if the user is requesting a new downloadable document.

    Strategy:
    1. Keyword gate      – instant reject if no doc-related words at all
    2. Shared-doc check  – if the message is clearly pasted/shared content,
                           skip the regex and go straight to the LLM so the
                           full context is considered (regex would false-positive
                           on phrases like "Output Management determines the PDF")
    3. Regex fast-path   – instant accept for short, explicit generation commands
    4. LLM classifier    – holistic intent check for everything else
    """
    if not _DOC_KEYWORD_GATE.search(message):
        return None

    shared = _looks_like_shared_document(message)

    if not shared:
        regex_result = _detect_doc_intent_regex(message)
        if regex_result:
            return regex_result

    if chat_agent is None:
        return None

    classify_prompt = (
        "You are a single-purpose JSON classifier that decides whether a user wants "
        "you to generate a new downloadable document file.\n\n"
        "CRITICAL DISTINCTION — read carefully:\n"
        "  A) The user is SHARING content (a spec, report, email, article, requirements "
        "document, etc.) and asking you to analyse, summarise, explain, or discuss it. "
        "     → doc=false\n"
        "  B) The user is ASKING you to CREATE a new file they can download. "
        "     → doc=true\n\n"
        "Key rules:\n"
        "- If the message is long and structured (multiple paragraphs, headings, tables) "
        "it is almost certainly case A — the user is sharing it, not requesting it.\n"
        "- Words like 'pdf', 'document', 'generate', 'output' that appear INSIDE shared "
        "content do NOT mean the user wants you to generate a document.\n"
        "- doc=true only when the user's OWN words (not the content they pasted) "
        "explicitly ask for a new file to download.\n"
        "- If in doubt, return doc=false.\n\n"
        f"Message: {json.dumps(message[:3000])}\n\n"
        "Reply with ONLY one JSON object, no other text:\n"
        '{"doc":true,"type":"word"}   <- user is asking for a Word/.docx file\n'
        '{"doc":true,"type":"pdf"}    <- user is asking for a PDF file\n'
        '{"doc":true,"type":"excel"}  <- user is asking for an Excel/spreadsheet\n'
        '{"doc":false}                <- user is sharing content or asking a question\n\n'
        "Examples:\n"
        '"give me a word document about X" -> {"doc":true,"type":"word"}\n'
        '"here is our spec, can you explain it?" -> {"doc":false}\n'
        '"[long functional specification text]" -> {"doc":false}\n'
        '"will generate the PDF" (internal process reference) -> {"doc":false}'
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

# ── Tool-call injection + detection ──────────────────────────────────────────
# When an app_id is present, registered tools are listed and a compact
# instruction block is prepended to the message. If the LLM decides to execute
# one it appends the exact JSON marker below. The streaming loop strips the
# marker before forwarding chunks and emits a tool_call SSE event instead.

_TOOL_CALL_MARKER = '{"__btp_tool_call__"'


def _try_extract_tool_call(text: str) -> Optional[dict]:
    """Parse a complete __btp_tool_call__ JSON object from the start of *text*.

    Returns the inner dict (tool_key, entity_key, parameters, confidence) on
    success, or None if the JSON is incomplete / malformed.
    """
    if not text.startswith(_TOOL_CALL_MARKER):
        return None
    depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[:i + 1])
                    return parsed.get('__btp_tool_call__') or None
                except Exception:
                    return None
    return None  # incomplete JSON — keep buffering


async def _load_app_tools(app_id: Optional[str]) -> list:
    """Fetch registered tools for *app_id* (best-effort, returns [] on failure)."""
    if not app_id:
        return []
    try:
        from app.db.session import get_optional_db
        from app.services.tool_catalog_service import list_tools
        _agen = get_optional_db()
        _sess = await _agen.__anext__()
        try:
            return await list_tools(_sess, app_id)
        finally:
            await _agen.aclose()
    except Exception:
        return []


def _build_tool_call_context(tools: list) -> str:
    """Build the system instruction block injected before the user message.

    Keeps the block compact so it doesn't crowd the context window.
    """
    if not tools:
        return ""
    lines = [
        "[REGISTERED TOOLS — use when the user wants to execute an action or function]",
    ]
    for t in tools:
        tt = t.tool_type.value if hasattr(t.tool_type, 'value') else str(t.tool_type)
        bind = (t.binding.value if t.binding and hasattr(t.binding, 'value') else (t.binding or 'unbound'))
        label = t.display_name or t.name or t.tool_key
        if t.parameters:
            pstr = ", ".join(
                f"{p.name}({'req' if p.required else 'opt'})"
                for p in t.parameters
            )
            lines.append(f"  • {t.tool_key} ({tt}, {bind}): {label} | params: {pstr}")
        else:
            lines.append(f"  • {t.tool_key} ({tt}, {bind}): {label}")
    lines += [
        "",
        "When the user's intent matches a tool, end your COMPLETE response with this JSON on its own line:",
        '{"__btp_tool_call__": {"tool_key": "TheKey", "entity_key": "ID_OR_OMIT", "parameters": {}, "confidence": 0.95}}',
        "Rules: set entity_key only for bound operations; include only parameters explicitly stated by the user; omit entity_key for unbound tools.",
        "[END REGISTERED TOOLS]",
        "",
    ]
    return "\n".join(lines)


def _format_exec_result(
    display_name: str,
    entity_key: Optional[str],
    result_data: object,
    messages: list,
) -> str:
    """Build a rich, client-facing confirmation for the chat bubble."""

    def _cap_words(s: str) -> str:
        """Capitalize the first letter of each word, leave the rest intact (preserves acronyms)."""
        return " ".join(w[:1].upper() + w[1:] for w in s.split() if w)

    # Readable action label: "FertilizerBlendService.FertilizerBlend.renderPDF" → "Render PDF"
    raw_action = display_name.split(".")[-1] if "." in display_name else display_name
    action_label = _cap_words(re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', raw_action).strip())

    # Primary reference fields — checked in order, first non-empty match is the headline number
    _PRIMARY = [
        "orderID", "OrderID", "orderNo", "formulaID", "formulationID",
        "ManufacturingOrder", "ProductionOrder", "ProcessOrder", "OrderNumber",
        "SalesOrder", "PurchaseOrder", "DeliveryOrder", "DeliveryDocument",
        "BillOfMaterial", "DocumentNumber", "InspectionLot", "Notification",
        "ID", "id",
    ]

    # Fields never shown (binary blobs, OData metadata, draft flags)
    _SKIP = {
        "fileContent", "@odata.context", "@odata.etag", "@odata.metadataEtag",
        "IsActiveEntity", "HasActiveEntity", "HasDraftEntity",
    }

    def _camel_label(s: str) -> str:
        """Convert camelCase/PascalCase to readable label, preserving acronyms.
        "orderID" → "Order ID",  "ManufacturingOrder" → "Manufacturing Order"
        """
        s = re.sub(r'(?<=[a-z])([A-Z])', r' \1', s)       # lowercase→UPPER boundary
        s = re.sub(r'(?<=[A-Z])([A-Z][a-z])', r' \1', s)  # UPPER→Upper boundary (acronyms)
        return _cap_words(s.strip())

    ref_label: str | None = None
    ref_value: str | None = None
    extra_fields: list = []

    if isinstance(result_data, dict):
        for key in _PRIMARY:
            val = result_data.get(key)
            if val and str(val) not in ("", "None", "null", "0", "false"):
                ref_label = _camel_label(key)
                ref_value = str(val)
                break

        for k, v in result_data.items():
            if k in _SKIP or k == (ref_label or ""):
                continue
            if isinstance(v, (dict, list)) or v is None:
                continue
            sv = str(v)
            if sv in ("", "None", "null") or len(sv) > 200:
                continue
            extra_fields.append((_camel_label(k), sv))
            if len(extra_fields) >= 4:
                break

    lines: list[str] = []

    if ref_label and ref_value and entity_key:
        lines.append(f"**{ref_label} #{ref_value}** created for Blend **{entity_key}**.")
    elif ref_label and ref_value:
        lines.append(f"**{ref_label} #{ref_value}** created successfully.")
    elif entity_key:
        lines.append(f"**{action_label}** completed for Blend **{entity_key}**.")
    else:
        raw = (messages[0] if messages else "Completed successfully.").split(".")[0]
        lines.append(f"**{action_label}** — {raw.strip()}.")

    if extra_fields:
        lines.append("")
        for label, val in extra_fields:
            lines.append(f"- **{label}**: {val}")

    if messages:
        lines.append("")
        for msg in messages[:3]:
            lines.append(f"> {msg.rstrip('.')}.")

    return "\n\n" + "\n".join(lines)


async def _execute_tool_inline(request, tool_call: dict):
    """Execute a tool the LLM decided to call and stream status chunks + tool_result event.

    Yields SSE strings directly so the caller can `yield` them into the stream.
    All errors are caught — the stream is never broken.

    Step pipeline (4 steps):
      1. analyzing  — broadcast before any I/O so the UI animates immediately
      2. found      — tool resolved from DB
      3. preparing  — entity key assembled, request prepared
      4. executing  — HTTP call in flight
    """
    if not request.app_id:
        return

    tool_key   = tool_call.get("tool_key", "")
    entity_key = tool_call.get("entity_key") or None
    parameters = tool_call.get("parameters") or {}

    _TOTAL = 4  # total pipeline steps shown in the UI

    def _status(step: str, step_num: int, **kwargs) -> str:
        return f"data: {json.dumps({'type': 'exec_status', 'step': step, 'step_num': step_num, 'total_steps': _TOTAL, **kwargs})}\n\n"

    def _chunk(text: str) -> str:
        return f"data: {json.dumps({'type': 'chunk', 'content': text})}\n\n"

    def _tool_result(data: dict) -> str:
        return f"data: {json.dumps({'type': 'tool_result', **data})}\n\n"

    # Step 1 — before any I/O so the animation panel appears immediately
    yield _status("analyzing", 1)

    odata_token  = request.odata_token or (request.fiori_context or {}).get("odata_token")
    display_name = tool_key

    try:
        from app.db.session import get_optional_db
        from app.services.tool_catalog_service import get_tool
        from app.services.action_execution.executor import ActionExecutionService
        from app.services.action_execution.models import ActionExecutionRequest

        _agen    = get_optional_db()
        _session = await _agen.__anext__()
        try:
            _tool_def = await get_tool(_session, request.app_id, tool_key)
            if _tool_def is not None:
                display_name = _tool_def.display_name or _tool_def.name or tool_key

            # Step 2 — tool identified
            yield _status("found", 2, tool=display_name)

            # Step 3 — request prepared (entity key resolved)
            entity_label = entity_key or "record"
            yield _status("preparing", 3, entity=entity_label)

            # Step 4 — HTTP call in flight
            yield _status("executing", 4)

            exec_req = ActionExecutionRequest(
                app_id=request.app_id,
                tool_key=tool_key,
                parameters=parameters,
                entity_key=entity_key,
                odata_token=odata_token,
            )
            result = await ActionExecutionService().execute(exec_req, session=_session)
        finally:
            await _agen.aclose()

        if result.success:
            res_data = result.result if isinstance(result.result, dict) else {}
            if res_data.get("executionType") == "UI_ACTION":
                msg = f"{display_name} triggered."
                yield _status("success", 4, message=msg)
                yield _chunk(f"\n✅ {msg}")
                yield _tool_result({
                    "success":        True,
                    "tool_key":       tool_key,
                    "execution_type": "UI_ACTION",
                    "frontend_event": res_data.get("frontendEvent"),
                    "payload":        res_data.get("payload", {}),
                })
            elif res_data.get("fileContent"):
                # Tool returned binary content (e.g. renderPDF via Adobe Lifecycle).
                # Store it and return a viewable URL instead of dumping base64 into the chat.
                import base64 as _b64
                from app.services.export_store import ExportStore as _ES
                from app.config import get_settings as _gs
                _cfg = _gs()
                _pdf_url: Optional[str] = None
                try:
                    _pdf_bytes = _b64.b64decode(res_data["fileContent"])
                    _safe = re.sub(r'[^a-z0-9_]', '_', (display_name.split(".")[-1] or "file").lower())
                    _key = _ES.put_raw(_pdf_bytes, "application/pdf", f"{_safe}_{entity_key or 'record'}.pdf")
                    _base = (_cfg.backend_base_url or "http://localhost:8000").rstrip("/")
                    _pdf_url = f"{_base}/api/export/{_key}/view"
                except Exception as _e:
                    logger.warning("[chat] PDF binary storage failed for %s: %s", tool_key, _e)

                msgs  = result.messages or []
                label = f"Blend **{entity_key}**" if entity_key else "the record"
                if _pdf_url:
                    yield _status("success", 4, message=f"PDF ready for {label.replace('**', '')}")
                    yield _chunk(
                        f"\n\nPDF generated for {label}.\n\n"
                        f"[View PDF]({_pdf_url}) &nbsp;·&nbsp; [Download PDF]({_pdf_url})\n\n"
                        f"_Link valid for 30 minutes._"
                    )
                    yield _tool_result({
                        "success":  True,
                        "tool_key": tool_key,
                        "messages": msgs,
                        "pdf_url":  _pdf_url,
                    })
                else:
                    # PDF storage failed — fall back to generic message
                    yield _status("success", 4, message="PDF generated.")
                    yield _chunk(f"\n\nPDF generated for {label}. (Preview unavailable — binary storage failed.)")
                    yield _tool_result({"success": True, "tool_key": tool_key, "messages": msgs})
            else:
                msgs    = result.messages or []
                summary = msgs[0] if msgs else "Completed successfully."
                rich_md = _format_exec_result(display_name, entity_key, result.result, msgs)
                yield _status("success", 4, message=summary)
                yield _chunk(rich_md)
                yield _tool_result({
                    "success":  True,
                    "tool_key": tool_key,
                    "messages": msgs,
                    "result":   result.result,
                })
        else:
            err = result.error.message if result.error else "Execution failed."
            yield _status("error", 4, message=err)
            yield _chunk(f"\n\n❌ **{display_name}** — {err}")
            yield _tool_result({"success": False, "tool_key": tool_key, "error": err})

    except Exception as exc:
        logger.error("[chat] Inline tool execution error tool='%s': %s", tool_key, exc, exc_info=True)
        err = str(exc)
        yield _status("error", 4, message=err)
        yield _chunk(f"\n\n❌ **{display_name}** failed: {err}")
        yield _tool_result({"success": False, "tool_key": tool_key, "error": err})


def _save_chat_to_db_sync(
    session_id: str,
    app_id: Optional[str],
    user_id: Optional[str],
    user_message: str,
    ai_response: str,
) -> None:
    """Persist chat session + both messages to Neon (chat_sessions / chat_messages tables).

    Runs in a thread-pool executor so it never blocks the async event loop.
    All DB errors are swallowed — message storage is best-effort and must not
    break the streaming response.
    """
    try:
        from app.api.apps import _neon_conn
        conn = _neon_conn()
        if not conn:
            return
        app_key = app_id or "__global__"
        try:
            with conn:
                with conn.cursor() as cur:
                    # 1. Ensure the application row exists (upsert)
                    cur.execute(
                        """
                        INSERT INTO applications (application_key, name)
                        VALUES (%s, %s)
                        ON CONFLICT (application_key)
                        DO UPDATE SET name = EXCLUDED.name
                        RETURNING id
                        """,
                        (app_key, app_key),
                    )
                    app_uuid = str(cur.fetchone()[0])

                    # 2. Upsert the chat session (idempotent on the same session_id)
                    cur.execute(
                        """
                        INSERT INTO chat_sessions
                            (id, application_id, user_id, user_subject, started_at)
                        VALUES (%s::uuid, %s::uuid, %s, %s, NOW())
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (session_id, app_uuid, user_id, user_id),
                    )

                    # 3. User message
                    cur.execute(
                        """
                        INSERT INTO chat_messages
                            (id, session_id, role, content, created_at)
                        VALUES (%s::uuid, %s::uuid, 'user', %s, NOW())
                        """,
                        (str(_uuid_module.uuid4()), session_id, user_message[:50_000]),
                    )

                    # 4. Assistant message
                    cur.execute(
                        """
                        INSERT INTO chat_messages
                            (id, session_id, role, content, created_at)
                        VALUES (%s::uuid, %s::uuid, 'assistant', %s, NOW())
                        """,
                        (str(_uuid_module.uuid4()), session_id, ai_response[:50_000]),
                    )
        except Exception as db_err:
            logger.warning("[chat_db] DB write error: %s", db_err)
        finally:
            conn.close()
    except Exception as e:
        logger.debug("[chat_db] Skipped (no Neon connection): %s", e)


def _enrich_message_with_fiori_context(message: str, fiori_context: dict | None) -> str:
    """Prepend a structured context block to the user message when a Fiori app has sent entity data.

    Handles both snake_case keys (from BTP Copilot SDK) and camelCase keys (legacy format).
    """
    if not fiori_context:
        return message

    # Support both snake_case (SDK widget) and camelCase (legacy) key formats
    app_id      = fiori_context.get("app_id") or fiori_context.get("appId")
    service_url = fiori_context.get("service_url") or fiori_context.get("serviceUrl")
    view        = fiori_context.get("current_view") or fiori_context.get("urlHash")
    entity      = fiori_context.get("entity_data") or fiori_context.get("entityData")
    extra       = fiori_context.get("extra") or {}
    schema_hint = extra.get("schema_hint") if isinstance(extra, dict) else None

    lines = ["[Context from the Fiori application you are embedded in]"]
    if app_id:
        lines.append(f"App: {app_id}")
    if service_url:
        lines.append(f"OData service: {service_url}")
    if view:
        lines.append(f"Current view: {view}")
    if entity and isinstance(entity, dict):
        lines.append("Current record:")
        for k, v in entity.items():
            if v is not None:
                lines.append(f"  {k}: {v}")
    if schema_hint:
        # Schema auto-fetched from $metadata by the widget — gives AI entity field knowledge
        lines.append(f"OData entity schema (from $metadata):\n{schema_hint[:3000]}")
    lines.append("[End of context]\n")
    return "\n".join(lines) + message


async def _generate_doc_event(message: str, doc_type: str, app_id: str | None) -> dict:
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

# Initialize agent router (singleton)
# The router dispatches to:
#   GlobalChatAgent   — when no app_id / fiori_context (standalone / global mode)
#   AppContextAgent   — when app_id or fiori_context is present (embedded Fiori mode)
#
# Supports unlimited CAP apps simultaneously — each app registers under its own
# app_id via POST /api/apps/register-service-tool. The service tool registry
# (persisted to service_tools.json) is keyed by app_id, so 10 different apps
# each get their own set of OData service URLs. No per-app config is needed here.
try:
    if settings.llm_provider == "sap_ai_core":
        logger.info("SAP AI Core mode — initialising AppContextAgent + GlobalChatAgent")
        if not all([settings.sap_aicore_url, settings.sap_aicore_client_id, settings.sap_aicore_client_secret]):
            raise ValueError(
                "SAP AI Core requires: SAP_AICORE_URL, SAP_AICORE_CLIENT_ID, SAP_AICORE_CLIENT_SECRET"
            )
        from app.agents.global_agent import GlobalChatAgent
        from app.agents.router import AgentRouter
        global_agent = GlobalChatAgent()
        app_agent = SAPAICoreAgent(
            url=settings.sap_aicore_url,
            client_id=settings.sap_aicore_client_id,
            client_secret=settings.sap_aicore_client_secret,
            model_id=settings.sap_aicore_model_id,
            deployment_id=settings.sap_aicore_deployment_id,
            auth_url=settings.sap_aicore_auth_url,
        )
        chat_agent = AgentRouter(global_agent=global_agent, app_agent=app_agent)
    else:
        logger.info("OpenAI mode — initialising AppContextAgent + GlobalChatAgent")
        from app.agents.global_agent import GlobalChatAgent
        from app.agents.router import AgentRouter
        global_agent = GlobalChatAgent()
        app_agent = ChatAgent()
        chat_agent = AgentRouter(global_agent=global_agent, app_agent=app_agent)

    logger.info("Chat agent router initialised successfully")
except Exception as e:
    logger.error(f"Failed to initialise chat agent router: {e}")
    chat_agent = None


# ── New context pipeline (feature-flagged) ────────────────────────────────────
# When ENABLE_CONTEXT_PIPELINE is on, a chat turn's context is prepared by the
# shared Planner → Retrieval Orchestrator → Context Builder pipeline and handed to
# the existing agent as `prepared_context`. Default off = unchanged legacy flow.
_CONTEXT_PIPELINE_ENABLED = getattr(settings, "enable_context_pipeline", False)


async def _build_prepared_context(request: "ChatRequest", *, user_id, session_id):
    """Run the shared context pipeline for one chat turn and return the prepared
    context string for the agents.

    Returns:
      * None  — flag off, agent unavailable, or ANY failure → caller uses the
                UNCHANGED legacy flow (agent does its own retrieval).
      * str   — the rendered context ("" means the pipeline ran but found nothing;
                agents then skip their own retrieval and inject no block).

    NEVER raises: a pipeline problem must never break streaming or a chat turn.
    Both the embedded-Fiori and Global chatbots go through this same function;
    only the ConversationContext (built by the mapper) differs.
    """
    if not _CONTEXT_PIPELINE_ENABLED or chat_agent is None:
        return None
    try:
        from app.db.session import get_optional_db
        from app.services.chat_context.mapper import chat_request_to_conversation_context
        from app.services.chat_context.pipeline import get_chat_pipeline

        cc = chat_request_to_conversation_context(request, user_id=user_id, session_id=session_id)
        # get_optional_db is an async-generator dependency; drive it manually because
        # we are outside FastAPI's Depends machinery (inside event_generator).
        _agen = get_optional_db()
        _session = await _agen.__anext__()
        try:
            out = await get_chat_pipeline().run(cc, session=_session)
        finally:
            await _agen.aclose()
        if out is None:
            return None
        logger.info(
            "[chat.sap_ai_core_request] channel=%s app_id=%s intent=%s confidence=%.3f "
            "retrievers=%s token_estimate=%d prepared_chars=%d pipeline_ms=%.1f",
            cc.channel.value, cc.app_id, out.intent, out.confidence,
            out.retrievers_used, out.token_estimate, len(out.prepared_context), out.total_ms,
        )
        return out.prepared_context
    except Exception as e:
        logger.error(
            f"[chat.pipeline] context pipeline failed — falling back to legacy flow: {e}",
            exc_info=True,
        )
        return None


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest, http_request: Request, current_user=Depends(get_current_user)) -> ChatResponse:
    """Send a message and get a response (requires auth)"""
    if chat_agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat service not available")

    # Extract the authenticated user's identity from the verified JWT.
    # Prefer user_name (XSUAA) → email → sub (generic JWT sub claim).
    user_id: str | None = (
        current_user.get("user_name")
        or current_user.get("email")
        or current_user.get("sub")
    )

    try:
        history = None
        if request.conversation_history:
            history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]

        _backend_url = str(http_request.base_url).rstrip("/")
        # New pipeline (flag-gated). None => unchanged legacy flow.
        prepared_context = await _build_prepared_context(
            request, user_id=user_id, session_id=request.session_id,
        )
        result = await chat_agent.get_response(
            message=request.message,
            history=history,
            app_id=request.app_id,
            fiori_context=request.fiori_context,
            odata_token=getattr(request, 'odata_token', None),
            user_id=user_id,
            backend_url=_backend_url,
            prepared_context=prepared_context,
        )

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
async def chat_stream(request: ChatRequest, http_request: Request, current_user=Depends(get_current_user)):
    if chat_agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat service not available")

    # Extract authenticated user identity from the verified JWT.
    _user_id: str | None = (
        current_user.get("user_name")
        or current_user.get("email")
        or current_user.get("sub")
    )

    _backend_url = str(http_request.base_url).rstrip("/")

    async def event_generator():
        # Use caller-supplied session_id or mint a new one for this conversation turn.
        _session_id = request.session_id or str(_uuid_module.uuid4())
        _response_buffer: list = []

        try:
            history = None
            if request.conversation_history:
                history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]

            start_time = time.time()
            enriched_message = _enrich_message_with_fiori_context(request.message, request.fiori_context)

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

                    # Build additional context from conversation history so the LLM
                    # has the live OData data that was shown in the previous turn.
                    _doc_ctx_parts = []
                    if history:
                        for _hm in history[-8:]:
                            _role = _hm.get("role", "user").upper()
                            _content = (_hm.get("content") or "")[:3000]
                            _doc_ctx_parts.append(f"{_role}: {_content}")
                    # Include current screen entity data (e.g. orderID=2466)
                    _entity_data = (request.fiori_context or {}).get("entity_data") or {}
                    if _entity_data:
                        _doc_ctx_parts.append(
                            f"CURRENT SCREEN RECORD: {json.dumps(_entity_data)[:800]}"
                        )
                    _doc_additional_context = "\n\n".join(_doc_ctx_parts) or None

                    data = await _get_document_content(request.message, doc_type, _doc_additional_context)

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
                # Inject tool-call instructions when an app_id is present.
                if request.app_id:
                    _app_tools = await _load_app_tools(request.app_id)
                    _tool_ctx = _build_tool_call_context(_app_tools)
                    if _tool_ctx:
                        enriched_message = _tool_ctx + enriched_message

                # New pipeline (flag-gated, runs ONCE before the stream loop). None =>
                # legacy flow untouched; the agent does its own retrieval as before.
                prepared_context = await _build_prepared_context(
                    request, user_id=_user_id, session_id=_session_id,
                )

                # Lookahead buffer: hold back up to len(_TOOL_CALL_MARKER) chars so
                # the JSON marker never leaks to the frontend as visible text.
                _hold = ""
                _tool_call_result: Optional[dict] = None
                _LOOKAHEAD = len(_TOOL_CALL_MARKER)

                if hasattr(chat_agent, 'stream_response'):
                    async for chunk in chat_agent.stream_response(
                        message=enriched_message,
                        history=history,
                        app_id=request.app_id,
                        fiori_context=request.fiori_context,
                        odata_token=getattr(request, 'odata_token', None),
                        user_id=_user_id,
                        raw_message=request.message,
                        backend_url=_backend_url,
                        prepared_context=prepared_context,
                    ):
                        _response_buffer.append(chunk)
                        _hold += chunk

                        marker_pos = _hold.find(_TOOL_CALL_MARKER)
                        if marker_pos >= 0:
                            # Emit everything before the marker
                            if marker_pos > 0:
                                yield f"data: {json.dumps({'type': 'chunk', 'content': _hold[:marker_pos]})}\n\n"
                            tail = _hold[marker_pos:]
                            tc = _try_extract_tool_call(tail)
                            if tc is not None:
                                _tool_call_result = tc
                                _hold = ""
                            else:
                                _hold = tail  # incomplete JSON — keep buffering
                        else:
                            # Emit bytes more than LOOKAHEAD chars old (safe zone)
                            safe = len(_hold) - _LOOKAHEAD
                            if safe > 0:
                                yield f"data: {json.dumps({'type': 'chunk', 'content': _hold[:safe]})}\n\n"
                                _hold = _hold[safe:]
                else:
                    result = await chat_agent.get_response(
                        message=enriched_message,
                        history=history,
                        app_id=request.app_id,
                        fiori_context=request.fiori_context,
                        odata_token=getattr(request, 'odata_token', None),
                        user_id=_user_id,
                        raw_message=request.message,
                        backend_url=_backend_url,
                        prepared_context=prepared_context,
                    )
                    words = result["response"].split(" ")
                    for i, word in enumerate(words):
                        chunk = word if i == 0 else ' ' + word
                        _response_buffer.append(chunk)
                        _hold += chunk
                        await asyncio.sleep(0.03)
                    # For non-streaming path: detect tool_call in accumulated hold buffer
                    marker_pos = _hold.find(_TOOL_CALL_MARKER)
                    if marker_pos >= 0:
                        if marker_pos > 0:
                            yield f"data: {json.dumps({'type': 'chunk', 'content': _hold[:marker_pos]})}\n\n"
                        tc = _try_extract_tool_call(_hold[marker_pos:])
                        if tc is not None:
                            _tool_call_result = tc
                            _hold = ""
                    else:
                        if _hold:
                            yield f"data: {json.dumps({'type': 'chunk', 'content': _hold})}\n\n"
                            _hold = ""

                # Emit any remaining non-tool-call text still in the hold buffer
                if _hold and not _tool_call_result:
                    marker_pos = _hold.find(_TOOL_CALL_MARKER)
                    if marker_pos > 0:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': _hold[:marker_pos]})}\n\n"
                    elif marker_pos < 0:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': _hold})}\n\n"

                # Execute the tool the LLM decided to call, streaming status inline.
                # For apps without an app_id, fall back to the raw tool_call event.
                if _tool_call_result:
                    if request.app_id:
                        async for _ev in _execute_tool_inline(request, _tool_call_result):
                            yield _ev
                    else:
                        yield f"data: {json.dumps({'type': 'tool_call', **_tool_call_result})}\n\n"

                response_time = time.time() - start_time
                if prepared_context is not None:
                    logger.info(
                        "[chat.streaming_complete] session_id=%s app_id=%s chunks=%d "
                        "response_time=%.2f model=%s pipeline=on",
                        _session_id, request.app_id, len(_response_buffer),
                        response_time, model_name,
                    )
                yield f"data: {json.dumps({'type': 'done', 'model': model_name, 'response_time': round(response_time, 2), 'session_id': _session_id})}\n\n"

                # ── Persist session + messages to Neon DB (best-effort, non-blocking) ─
                _ai_text = "".join(_response_buffer)
                if _ai_text:
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        _save_chat_to_db_sync,
                        _session_id,
                        request.app_id,
                        _user_id,
                        request.message,
                        _ai_text,
                    )

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