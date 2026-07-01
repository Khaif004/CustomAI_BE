"""chat_context — the orchestration glue that runs the existing
Planner → Retrieval Orchestrator → Context Builder pipeline for the chat flow.

This package introduces NO new business logic. It only:
  * maps the transport `ChatRequest` to a channel-agnostic `ConversationContext`
    (`mapper`),
  * runs the existing, tested pipeline components and logs each stage
    (`pipeline.ChatPipelineService`),
  * renders the structured `LLMContext` into the plain-text context block the
    existing agents already consume (`renderer`) — a TEMPORARY adapter that a
    real Prompt Builder will replace.

It is consumed only by `app/api/chat.py`, behind a feature flag.
"""
from __future__ import annotations

from app.services.chat_context.pipeline import ChatPipelineService, PipelineOutput, get_chat_pipeline

__all__ = ["ChatPipelineService", "PipelineOutput", "get_chat_pipeline"]
