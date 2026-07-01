"""ConversationContext — the channel-agnostic input to the chat context pipeline.

It is deliberately INDEPENDENT of `ChatRequest` and of any Fiori/UI5 payload
shape. Both the embedded-Fiori chatbot and the Global chatbot produce a
`ConversationContext`; the ONLY difference between them is the data it carries
(see `channel`). The Planner and retrieval layers receive only primitive fields
they already accept (message, app_id, an opaque fiori_context dict, odata_token,
user_id, session) — the Planner never branches on Fiori-specific objects.

All fields except `message` are optional; the pipeline continues gracefully when
any are absent.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class Channel(str, Enum):
    """Which front-end produced the turn. Affects DATA only, not pipeline logic."""

    GLOBAL = "global"            # Global chatbot — no application context
    EMBEDDED_FIORI = "embedded"  # Embedded Fiori chatbot — app_id / fiori_context present


class ConversationContext(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")

    message: str                                   # the raw user query
    channel: Channel = Channel.GLOBAL

    app_id: Optional[str] = None
    odata_token: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Optional UI/Fiori bits. `fiori_context` is an OPAQUE dict passed straight to
    # the existing planner/retrieval inputs (which already read it defensively);
    # the rest are convenience fields for logging + a future UI-context retriever.
    # The Planner does not depend on any of these.
    fiori_context: Optional[Dict[str, Any]] = None
    current_entity: Optional[str] = None
    current_record: Optional[Dict[str, Any]] = None
    current_view: Optional[str] = None
    ui_context: Optional[Dict[str, Any]] = None

    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

    @property
    def is_app_context(self) -> bool:
        return bool(self.app_id) or bool(self.fiori_context)
