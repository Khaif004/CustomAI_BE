"""MemoryRetriever — placeholder for a future Conversation Memory engine.

There is no server-side conversation-history reader yet (history is supplied
client-side via ChatRequest.conversation_history). This retriever is wired into
the orchestration registry so CONVERSATION_MEMORY sources are handled uniformly,
but it returns an empty section today. A future implementation can read
chat_messages/chat_sessions without any change to the orchestrator.
"""
from __future__ import annotations

from typing import ClassVar

from app.models.planner import RetrievalSource
from app.services.retrieval.base import Retriever
from app.services.retrieval.models import RetrievalRequest, RetrieverResult, Section


class MemoryRetriever(Retriever):
    source: ClassVar[RetrievalSource] = RetrievalSource.CONVERSATION_MEMORY
    section: ClassVar[Section] = Section.CONVERSATION_MEMORY

    async def retrieve(self, request: RetrievalRequest) -> RetrieverResult:
        return self._empty()
