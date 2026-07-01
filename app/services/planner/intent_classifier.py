"""IntentClassifier — deterministic, rule-based primary-intent classification.

NO LLM. NO network. NO DB. Pure function over `PlanSignals`.

Approach: each intent has a scorer that accumulates weighted lexical/context
signals into a raw score in [0, 1]. We then pick the highest-scoring intent and
derive a margin-aware confidence so ambiguous inputs (two intents tied) yield
lower confidence than a single dominant intent.

The classifier deliberately uses VERB MOOD as the DATA_QUERY-vs-TOOL_EXECUTION
discriminator (read verbs → DATA_QUERY, mutating verbs → TOOL_EXECUTION). It runs
BEFORE tool/entity resolution; `PlannerService` applies a single bounded
post-resolution adjustment (e.g. a confirmed tool hit raises TOOL_EXECUTION).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from app.models.planner import Intent
from app.services.planner import text_signals as ts
from app.services.planner.text_signals import PlanSignals


@dataclass
class IntentScore:
    intent: Intent
    confidence: float
    scores: Dict[Intent, float] = field(default_factory=dict)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class IntentClassifier:
    """Stateless, reusable. Construct once; `classify` is pure."""

    def classify(self, signals: PlanSignals) -> IntentScore:
        scores = self._score_all(signals)
        return self.finalize(scores)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_all(self, s: PlanSignals) -> Dict[Intent, float]:
        norm = s.norm
        toks = s.tokens

        schema_tokens = ts.any_token(toks, ts.SCHEMA_TOKENS)
        schema_phrases = ts.any_phrase(norm, ts.SCHEMA_PHRASES)
        schema_present = bool(schema_tokens or schema_phrases)

        read_verbs = ts.matched_read_verbs(toks)
        # Imperative mood: the *leading* command verb is a mutating verb. This is
        # what distinguishes "create an order" (TOOL_EXECUTION) from
        # "show me all process orders" (DATA_QUERY) where a mutating-looking word
        # ("process") appears only as a noun later in the sentence.
        command_token = ts.first_command_token(toks)
        is_imperative_mut = command_token in ts.MUTATING_VERBS

        count_cue = (
            bool(ts.any_phrase(norm, ts.COUNT_PHRASES))
            or bool(ts.any_token(toks, ts.COUNT_TOKENS))
            or ts.AGG_RE.search(norm) is not None
            or ts.MY_INTENT_RE.search(norm) is not None
        )

        knowledge_phrases = ts.any_phrase(norm, ts.KNOWLEDGE_PHRASES)
        nav_phrases = ts.any_phrase(norm, ts.NAVIGATION_PHRASES)
        nav_tokens = ts.any_token(toks, ts.NAVIGATION_TOKENS)
        doc_phrases = ts.any_phrase(norm, ts.DOC_PHRASES)
        doc_artifact = ts.any_token(toks, ts.DOC_ARTIFACT_TOKENS)
        doc_verbs = ts.any_token(toks, ts.DOC_VERBS)
        code_tokens = ts.any_token(toks, ts.CODE_TOKENS)
        code_phrases = ts.any_phrase(norm, ts.CODE_PHRASES)
        code_file = ts.CODE_FILE_RE.search(s.raw) is not None
        greetings = ts.any_phrase(norm, ts.GREETING_PHRASES)

        scores: Dict[Intent, float] = {i: 0.0 for i in Intent}

        # ── DATA_QUERY ──────────────────────────────────────────────────────
        dq = 0.0
        if not schema_present:
            if s.primary_noun:
                dq += ts.STRONG
            elif read_verbs:
                dq += ts.MEDIUM
        if count_cue:
            dq += ts.MEDIUM
        if s.has_entity_data or s.has_service_url:
            dq += ts.WEAK
        scores[Intent.DATA_QUERY] = _clamp(dq)

        # ── TOOL_EXECUTION ──────────────────────────────────────────────────
        te = 0.0
        if is_imperative_mut:
            te += ts.STRONG
            if s.primary_noun or s.content_tokens:
                te += ts.MEDIUM
            if s.has_app_context:
                te += ts.WEAK
        scores[Intent.TOOL_EXECUTION] = _clamp(te)

        # ── SCHEMA ──────────────────────────────────────────────────────────
        sc = 0.0
        if schema_present:
            sc += ts.STRONG
            if s.has_schema_hint:
                sc += ts.WEAK
        scores[Intent.SCHEMA] = _clamp(sc)

        # ── KNOWLEDGE ───────────────────────────────────────────────────────
        kn = 0.0
        if knowledge_phrases:
            kn += ts.STRONG
            if not s.has_app_context:
                kn += ts.WEAK
        scores[Intent.KNOWLEDGE] = _clamp(kn)

        # ── NAVIGATION ──────────────────────────────────────────────────────
        nv = 0.0
        nav_present = bool(nav_phrases) or (bool(nav_tokens) and (read_verbs or "open" in toks or "go" in toks))
        if nav_present:
            nv += ts.STRONG if s.has_app_context else ts.WEAK
        scores[Intent.NAVIGATION] = _clamp(nv)

        # ── DOCUMENTATION (artifact) ────────────────────────────────────────
        dc = 0.0
        if doc_verbs and doc_artifact:
            dc += ts.STRONG
        elif doc_phrases:
            dc += ts.MEDIUM
        elif doc_artifact:
            dc += ts.MEDIUM
        scores[Intent.DOCUMENTATION] = _clamp(dc)

        # ── CODE_INTELLIGENCE ───────────────────────────────────────────────
        ci = 0.0
        if code_tokens or code_phrases:
            ci += ts.STRONG
        if code_file:
            ci += ts.WEAK
        scores[Intent.CODE_INTELLIGENCE] = _clamp(ci)

        # ── GENERAL_CHAT (fallback floor) ───────────────────────────────────
        gc = ts.GENERAL_FLOOR
        if greetings:
            gc += ts.MEDIUM
        if not s.has_app_context:
            gc += ts.WEAK
        scores[Intent.GENERAL_CHAT] = _clamp(gc)

        return scores

    # ── Finalization (re-usable after post-resolution adjustment) ─────────────

    def finalize(self, scores: Dict[Intent, float]) -> IntentScore:
        """Pick the top intent and compute a margin-aware confidence in [0, 0.99].

        confidence = top + (1 - top) * 0.5 * margin, where
        margin = (top - second) / top. A dominant intent (large margin) pushes
        confidence well above its raw score; a tie (margin≈0) keeps confidence at
        the raw score, signaling ambiguity. Capped below 1.0 — the Planner is
        rule-based, never certain.
        """
        if not scores:
            return IntentScore(intent=Intent.GENERAL_CHAT, confidence=ts.GENERAL_FLOOR, scores={})

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_intent, top = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0

        margin = (top - second) / top if top > 0 else 0.0
        confidence = _clamp(top + (1.0 - top) * 0.5 * margin, 0.0, 0.99)
        return IntentScore(intent=top_intent, confidence=round(confidence, 4), scores=scores)
