"""Token estimation for the Context Builder.

Model-agnostic by design: a `TokenEstimator` is injected into the builder, so a
future precise tokenizer (tiktoken, a model-specific counter, etc.) can replace
the default WITHOUT changing the builder. The default is a dependency-free
heuristic (~4 chars/token) — good enough for budgeting and identical across
providers, which keeps the Context Builder free of any model-specific code.
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenEstimator(Protocol):
    def estimate(self, text: str) -> int: ...


class HeuristicTokenEstimator:
    """Provider-neutral estimate: ceil(len(text) / chars_per_token)."""

    def __init__(self, chars_per_token: float = 4.0):
        self._cpt = chars_per_token if chars_per_token > 0 else 4.0

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self._cpt))


# Process-wide default instance (stateless; safe to share).
DEFAULT_TOKEN_ESTIMATOR: TokenEstimator = HeuristicTokenEstimator()
