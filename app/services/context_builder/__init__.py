"""Context Builder package.

Transforms a `RetrievalContext` into a structured, model-agnostic `LLMContext`.
It consumes ONLY a RetrievalContext — it never calls a retriever, the Planner,
or any LLM, and builds no prompts. It is independent of the Chat API.

Exposes stateless DI providers (mirroring the planner/retrieval packages):
  * `get_context_builder_settings()` — the configurable token budgets.
  * `get_context_builder()` — the process-wide stateless ContextBuilder.
"""
from __future__ import annotations

from functools import lru_cache

from app.models.llm_context import ContextBuilderSettings
from app.services.context_builder.builder import ContextBuilder
from app.services.context_builder.tokenizer import DEFAULT_TOKEN_ESTIMATOR

__all__ = ["ContextBuilder", "get_context_builder", "get_context_builder_settings"]


@lru_cache(maxsize=1)
def get_context_builder_settings() -> ContextBuilderSettings:
    """Default token-budget settings. Override via FastAPI dependency_overrides or
    by passing a settings instance into `ContextBuilder.build(context, settings=...)`."""
    return ContextBuilderSettings()


@lru_cache(maxsize=1)
def get_context_builder() -> ContextBuilder:
    """Process-wide stateless ContextBuilder with default settings + token estimator."""
    return ContextBuilder(
        settings=get_context_builder_settings(),
        tokenizer=DEFAULT_TOKEN_ESTIMATOR,
    )
