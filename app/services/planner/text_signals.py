"""Shared, pure text-signal utilities + vocabulary for the Planner.

Single source of truth for the Planner's deterministic lexical analysis. All
functions here are PURE (no I/O, no LLM, no network, no global state), so the
IntentClassifier, EntityResolver and ToolResolver share one vocabulary and the
unit tests can target these helpers directly.

The regexes/scoring here are adapted (copied) from the deterministic, non-LLM
helpers in `app/agents/sap_ai_core_agent.py` (e.g. the `_primary_noun` lead-in
regex, `_ent_slug`, `_trigrams`, the `_NOISE` stopword set). We copy rather than
import because those are private methods on a 2,900-line agent class; copying the
pure logic keeps the Planner self-contained and avoids modifying existing code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# ── Confidence weights (tunable) ──────────────────────────────────────────────
STRONG = 0.45
MEDIUM = 0.25
WEAK = 0.10
GENERAL_FLOOR = 0.30  # baseline so GENERAL_CHAT always has a small score

# ── Vocabulary ────────────────────────────────────────────────────────────────
# Read/retrieval verbs (single tokens). Phrase cues handled separately below.
READ_VERBS: Set[str] = {
    "show", "list", "get", "fetch", "find", "display", "view", "see",
    "retrieve", "lookup", "search", "browse",
}

# Mutating/imperative verbs that imply DOING something (TOOL_EXECUTION).
# NOTE: doc-generation verbs (generate/export/download/produce) are deliberately
# NOT here — they belong to DOCUMENTATION (see DOC_VERBS) and would otherwise
# steal "generate a pdf report" into TOOL_EXECUTION.
MUTATING_VERBS: Set[str] = {
    "create", "add", "update", "modify", "change", "edit", "delete", "remove",
    "submit", "approve", "reject", "cancel", "trigger", "run", "execute",
    "send", "post", "release", "set", "assign", "confirm", "close", "book",
    "schedule", "process", "complete", "reopen", "duplicate", "activate",
    "deactivate", "archive",
}

# Polite/pronoun/article lead words skipped when finding the leading command verb
# (so "can you please create an order" still reads as imperative "create").
LEAD_SKIP: Set[str] = {
    "please", "pls", "kindly", "can", "could", "would", "will", "you", "i",
    "we", "to", "let", "lets", "help", "need", "want", "like", "someone",
    "also", "just", "now", "then", "go", "ahead", "and", "the", "a", "an",
    "all", "me", "my", "for",
}

# Count / aggregation cues (phrases checked with phrase matcher).
COUNT_PHRASES: List[str] = [
    "how many", "number of", "count of", "total number",
]
COUNT_TOKENS: Set[str] = {"count", "total", "sum", "average", "avg", "min", "max"}
AGG_RE = re.compile(r"\b(?:by|per|group(?:ed)?\s+by|breakdown\s+by)\s+(\w+)", re.IGNORECASE)
MY_INTENT_RE = re.compile(
    r"\b(my|mine|i\s+have|i\s+created|i\s+made|assigned\s+to\s+me|my\s+own)\b",
    re.IGNORECASE,
)

# Schema / metadata vocabulary.
SCHEMA_TOKENS: Set[str] = {
    "schema", "field", "fields", "column", "columns", "attribute", "attributes",
    "property", "properties", "metadata", "association", "associations",
    "structure", "datatype", "datatypes",
}
SCHEMA_PHRASES: List[str] = [
    "data model", "entity type", "$metadata", "relationship between",
    "what fields", "which fields", "structure of", "data type",
]

# Knowledge / how-to vocabulary (phrases — must be specific to avoid eating
# DATA_QUERY/SCHEMA which also start with "what").
KNOWLEDGE_PHRASES: List[str] = [
    "what is", "what are the steps", "explain", "how do i", "how do you",
    "how to", "why is", "why does", "why do", "best practice", "best practices",
    "meaning of", "definition of", "what does it mean", "overview of",
    "difference between", "tell me about", "what's the purpose", "purpose of",
]

# Navigation vocabulary (phrases).
NAVIGATION_PHRASES: List[str] = [
    "go to", "navigate to", "navigate", "take me to", "open the", "open ",
    "show me the page", "where is the", "where can i find", "jump to",
    "switch to", "bring up",
]
NAVIGATION_TOKENS: Set[str] = {"page", "screen", "view", "tab", "section", "route"}

# Documentation-artifact vocabulary (report/export/document generation).
DOC_ARTIFACT_TOKENS: Set[str] = {"pdf", "word", "excel", "report", "spreadsheet", "docx"}
DOC_VERBS: Set[str] = {"generate", "export", "download", "produce"}
DOC_PHRASES: List[str] = [
    "documentation", "user guide", "user manual", "reference guide",
    "generate a report", "create a document", "export to",
]

# Code-intelligence vocabulary.
CODE_TOKENS: Set[str] = {
    "code", "function", "class", "method", "endpoint", "implementation",
    "handler", "module", "import", "imports", "cds", "exception", "traceback",
    "stacktrace", "repository", "repo", "snippet", "compile", "syntax",
}
CODE_PHRASES: List[str] = [
    "where is", "defined in", "call site", "source code", "how is it implemented",
    "stack trace", "which file",
]
CODE_FILE_RE = re.compile(r"\.\w{1,4}\b|`[^`]+`|\b\w+\.(py|ts|js|cds|java|sql)\b", re.IGNORECASE)

# Greeting / chitchat vocabulary.
GREETING_PHRASES: List[str] = [
    "hi", "hello", "hey", "thanks", "thank you", "who are you",
    "what can you do", "good morning", "good afternoon", "help me",
]

# Stopwords excluded from entity/tool scoring (copied from sap_ai_core_agent _NOISE).
NOISE: Set[str] = {
    "are", "be", "been", "can", "could", "did", "do", "does", "get",
    "give", "has", "have", "how", "in", "is", "it", "its", "list",
    "make", "many", "me", "of", "our", "show", "tell", "that", "the",
    "them", "there", "these", "this", "those", "to", "total", "us",
    "was", "were", "what", "which", "who", "will", "with", "would",
    "all", "any", "each", "fetch", "find", "from", "give", "their",
    "a", "an", "and", "or", "for", "on", "by", "my",
}

# Lead-in retrieval-verb regex (copied/condensed from sap_ai_core_agent _pn_match).
_PRIMARY_NOUN_RE = re.compile(
    r"\b(?:what\s+(?:are\s+)?(?:the\s+)?"
    r"|show\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?"
    r"|list\s+(?:all\s+)?(?:the\s+)?"
    r"|get\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?"
    r"|fetch\s+(?:all\s+)?(?:the\s+)?"
    r"|find\s+(?:all\s+)?(?:the\s+)?"
    r"|give\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?"
    r"|need\s+(?:all\s+)?(?:the\s+)?"
    r"|how\s+many\s+(?:the\s+)?"
    r")\s*(\w+)",
    re.IGNORECASE,
)


# ── Pure helpers ──────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokens(norm_text: str, *, drop_noise: bool = False) -> List[str]:
    """Split a normalized string into word tokens. Optionally drop stopwords."""
    raw = re.findall(r"[a-z0-9_]+", norm_text)
    if drop_noise:
        return [t for t in raw if t not in NOISE]
    return raw


def trigrams(s: str) -> Set[str]:
    """Trigram set for typo-tolerant similarity (copied from sap_ai_core_agent)."""
    s = re.sub(r"\W+", "", s.lower())
    return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def trigram_similarity(a: str, b: str) -> float:
    """Jaccard similarity of trigram sets, 0..1."""
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def ent_slug(entity: str) -> str:
    """camelCase/PascalCase -> hyphenated lowercase (copied from sap_ai_core_agent)."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "-", entity).lower()


def singular(word: str) -> str:
    """Very small English singularizer (orders -> order, entities -> entity)."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("ses") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def stem(word: str) -> str:
    """4-char prefix stem with light plural/-tion normalization (entity scoring)."""
    w = singular(word.lower())
    if w.endswith("tion") and len(w) > 5:
        w = w[:-4]
    return w[:4] if len(w) > 4 else w


def primary_noun(message: str) -> Optional[str]:
    """Extract the main subject noun after a retrieval lead-in verb, else None.

    Discards a captured stopword/copula (e.g. "what IS a process order" → None)
    so it doesn't masquerade as a data-query subject.
    """
    m = _PRIMARY_NOUN_RE.search(message or "")
    if not m:
        return None
    w = m.group(1).lower()
    return None if w in NOISE else w


def first_command_token(token_list: List[str]) -> Optional[str]:
    """Return the first token that isn't a polite/article/pronoun lead word —
    the imperative command verb candidate. None if the list is all lead words."""
    for t in token_list:
        if t not in LEAD_SKIP:
            return t
    return None


def contains_phrase(norm_text: str, phrase: str) -> bool:
    """Word-boundary-aware phrase containment on a normalized string."""
    p = phrase.strip().lower()
    if " " not in p and not p.endswith(" "):
        return re.search(rf"\b{re.escape(p)}\b", norm_text) is not None
    return p in norm_text


def any_phrase(norm_text: str, phrases: List[str]) -> List[str]:
    """Return the phrases that are present in the normalized text."""
    return [p for p in phrases if contains_phrase(norm_text, p)]


def any_token(token_list: List[str], vocab: Set[str]) -> Set[str]:
    """Return the tokens present in a vocabulary set."""
    return {t for t in token_list if t in vocab}


@dataclass(frozen=True)
class PlanSignals:
    """Pre-computed lexical + context signals shared across Planner components."""

    raw: str
    norm: str
    tokens: List[str] = field(default_factory=list)
    content_tokens: List[str] = field(default_factory=list)  # NOISE-filtered
    primary_noun: Optional[str] = None
    has_app_context: bool = False
    has_entity_data: bool = False
    has_service_url: bool = False
    has_schema_hint: bool = False

    @staticmethod
    def build(
        message: str,
        *,
        app_id: Optional[str],
        fiori_context: Optional[Dict] = None,
    ) -> "PlanSignals":
        norm = normalize(message)
        fc = fiori_context or {}
        extra = fc.get("extra") if isinstance(fc.get("extra"), dict) else {}
        has_entity_data = bool(fc.get("entity_data") or fc.get("entityData"))
        has_service_url = bool(fc.get("service_url") or fc.get("serviceUrl"))
        has_schema_hint = bool(extra.get("schema_hint")) if extra else False
        return PlanSignals(
            raw=message or "",
            norm=norm,
            tokens=tokens(norm),
            content_tokens=tokens(norm, drop_noise=True),
            primary_noun=primary_noun(message or ""),
            has_app_context=bool(app_id) or bool(fiori_context),
            has_entity_data=has_entity_data,
            has_service_url=has_service_url,
            has_schema_hint=has_schema_hint,
        )


def matched_mutating_verbs(token_list: List[str]) -> Set[str]:
    return any_token(token_list, MUTATING_VERBS)


def matched_read_verbs(token_list: List[str]) -> Set[str]:
    return any_token(token_list, READ_VERBS)
