"""The concrete retrievers + the default retriever set.

`default_retrievers()` is the single place that lists which retrievers exist.
Adding a future retriever = add one line here; the orchestrator never changes
(open/closed).
"""
from __future__ import annotations

from typing import List

from app.services.retrieval.base import Retriever
from app.services.retrieval.retrievers.documentation_retriever import DocumentationRetriever
from app.services.retrieval.retrievers.keyword_retriever import KeywordRetriever
from app.services.retrieval.retrievers.live_odata_retriever import LiveODataRetriever
from app.services.retrieval.retrievers.memory_retriever import MemoryRetriever
from app.services.retrieval.retrievers.metadata_retriever import MetadataRetriever
from app.services.retrieval.retrievers.tool_retriever import ToolRetriever
from app.services.retrieval.retrievers.vector_retriever import VectorRetriever

__all__ = [
    "MetadataRetriever",
    "ToolRetriever",
    "VectorRetriever",
    "KeywordRetriever",
    "LiveODataRetriever",
    "MemoryRetriever",
    "DocumentationRetriever",
    "default_retrievers",
]


def default_retrievers() -> List[Retriever]:
    """The production retriever set, one per supported RetrievalSource."""
    return [
        MetadataRetriever(),
        ToolRetriever(),
        VectorRetriever(),
        KeywordRetriever(),
        LiveODataRetriever(),
        MemoryRetriever(),
        DocumentationRetriever(),
    ]
