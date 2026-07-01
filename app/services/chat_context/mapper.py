"""Map the transport contract (`ChatRequest`) to the channel-agnostic
`ConversationContext`. Pure/sync, no I/O.

This is the ONLY module that knows about both `ChatRequest` and
`ConversationContext`, so the pipeline stays transport-agnostic. The same mapper
serves both the embedded-Fiori and Global chatbots — only the resulting `channel`
and which optional fields are populated differ (driven entirely by the data).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.models.chat import ChatRequest
from app.models.conversation_context import Channel, ConversationContext


def _history(req: ChatRequest) -> List[Dict[str, str]]:
    if not req.conversation_history:
        return []
    return [{"role": m.role, "content": m.content} for m in req.conversation_history]


def _fc_get(fc: Optional[Dict[str, Any]], *keys: str):
    """Read the first present key from a fiori_context dict (snake + camel)."""
    if not fc:
        return None
    for k in keys:
        v = fc.get(k)
        if v not in (None, ""):
            return v
    return None


def _as_str(v) -> Optional[str]:
    return v if isinstance(v, str) else None


def _as_dict(v) -> Optional[Dict[str, Any]]:
    return v if isinstance(v, dict) else None


def chat_request_to_conversation_context(
    req: ChatRequest,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> ConversationContext:
    fc = req.fiori_context or None
    channel = Channel.EMBEDDED_FIORI if (req.app_id or fc) else Channel.GLOBAL
    return ConversationContext(
        message=req.message,
        channel=channel,
        app_id=req.app_id,
        odata_token=getattr(req, "odata_token", None),
        user_id=user_id,
        session_id=session_id,
        fiori_context=fc,
        # Best-effort UI extraction (graceful if absent); reads both casings. Type
        # guards keep an odd payload (e.g. a non-dict entity_data) from breaking
        # ConversationContext construction — the pipeline survives instead of
        # silently falling back to the legacy flow.
        current_entity=_as_str(_fc_get(fc, "current_entity", "currentEntity", "entity", "entityName")),
        current_record=_as_dict(_fc_get(fc, "entity_data", "entityData", "current_record", "currentRecord")),
        current_view=_as_str(_fc_get(fc, "current_view", "currentView", "urlHash")),
        ui_context=_as_dict(_fc_get(fc, "ui_context", "uiContext")),
        conversation_history=_history(req),
    )
