"""TEMPORARY adapter: render a structured `LLMContext` into the plain-text context
block the existing agents already consume (their `rag_context` / prepended
live-data string).

This is NOT a Prompt Builder. It does no model-specific formatting and no
instruction templating — it linearizes the already-prioritised, already-budgeted
`LLMContext` sections into text so the existing prompt-generation paths
(`ChatAgent._build_system_prompt`, `SAPAICoreAgent` message-prepend,
`GlobalChatAgent` history-prepend) work unchanged. Delete it when the real Prompt
Builder lands (requirement explicitly defers prompt redesign).
"""
from __future__ import annotations

import json
from typing import List

from app.models.llm_context import ContextItem, ContextSection, LLMContext

# Section render order + heading. Order mirrors the Context Builder's priority
# (exact business data first). SystemInstructions are intentionally NOT rendered
# here — they are structured directives for the future Prompt Builder, not prose.
_ORDER = [
    (ContextSection.LIVE_BUSINESS_DATA, "LIVE BUSINESS DATA (current, exact — prefer these numbers)"),
    (ContextSection.CURRENT_UI_CONTEXT, "CURRENT UI CONTEXT"),
    (ContextSection.APPLICATION_METADATA, "APPLICATION METADATA"),
    (ContextSection.TOOL_METADATA, "AVAILABLE TOOLS"),
    (ContextSection.DOCUMENTATION, "DOCUMENTATION"),
    (ContextSection.CONVERSATION_CONTEXT, "CONVERSATION CONTEXT"),
    (ContextSection.SEMANTIC_KNOWLEDGE, "RETRIEVED KNOWLEDGE"),
]


def _item_text(it: ContextItem) -> str:
    parts: List[str] = []
    if it.title:
        parts.append(it.title)
    if it.content:
        parts.append(it.content)
    if it.data:
        try:
            parts.append(json.dumps(it.data, default=str, ensure_ascii=False))
        except Exception:
            parts.append(str(it.data))
    return " — ".join(p for p in parts if p)


def render_llm_context(ctx: LLMContext) -> str:
    """Return a flat text block for the agents, or '' when nothing was retrieved.

    An empty string is a meaningful result ("pipeline ran, nothing to add"): the
    agents treat a non-None prepared context as "skip internal retrieval", and an
    empty one simply injects no block.
    """
    blocks: List[str] = []
    for section, heading in _ORDER:
        items = getattr(ctx, section.value, [])
        if not items:
            continue
        lines = [f"- {t}" for t in (_item_text(it) for it in items) if t]
        if lines:
            blocks.append(f"{heading}:\n" + "\n".join(lines))
    return "\n\n".join(blocks)
