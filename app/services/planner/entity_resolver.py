"""EntityResolver — resolve a target CAP entity from a user message.

NO LLM, NO network. Reuses the existing in-memory service-tool registry
(`app.api.apps._service_tool_registry`) as the candidate source, behind an
`EntityRegistry` port so unit tests can inject a fake without importing the app.

Resolution cascade (deterministic, adapted from the agent's non-LLM entity logic):
  1. Fiori context entity hint   (the record the user is literally looking at)
  2. Alias exact match           (entity_aliases, when present — DB-loaded apps)
  3. Primary-noun exact/singular / full-message containment
  4. Compound refinement         (SalesOrder -> SalesOrderItems)
  5. Fuzzy scoring               (stem overlap + trigram similarity)
  6. No-guess                    (returns name=None rather than fabricating)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Set

from app.services.planner import text_signals as ts


class EntityRegistry(Protocol):
    """Port over the metadata/entity registry. The real impl reads the in-memory
    service-tool registry; tests provide a fake with the same surface."""

    def get_entities(self, app_id: Optional[str]) -> List[str]: ...
    def get_aliases(self, app_id: Optional[str]) -> Dict[str, str]: ...
    def service_url_for(self, app_id: Optional[str], entity: str) -> Optional[str]: ...


class InMemoryEntityRegistry:
    """Concrete EntityRegistry backed by `app.api.apps._service_tool_registry`.

    Imports the global lazily inside each method to avoid an import cycle with
    `app.api.apps` (which pulls in the knowledge base, aiohttp, etc.).
    """

    def _entries(self, app_id: Optional[str]) -> List[dict]:
        if not app_id:
            return []
        try:
            from app.api.apps import _service_tool_registry
        except Exception:
            return []
        return list(_service_tool_registry.get(app_id, []) or [])

    def get_entities(self, app_id: Optional[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for entry in self._entries(app_id):
            names = list(entry.get("entities", []) or [])
            names += list((entry.get("entity_fields", {}) or {}).keys())
            for n in names:
                short = n.split(".")[-1]  # registries sometimes store FQ names
                key = short.lower()
                if short and key not in seen:
                    seen.add(key)
                    out.append(short)
        return out

    def get_aliases(self, app_id: Optional[str]) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        for entry in self._entries(app_id):
            for alias, canonical in (entry.get("entity_aliases", {}) or {}).items():
                merged[str(alias).lower()] = str(canonical).split(".")[-1]
        return merged

    def service_url_for(self, app_id: Optional[str], entity: str) -> Optional[str]:
        """Pick the service whose URL slug best 'homes' the entity (home-score),
        among the services that expose it — mirrors the agent's _home_score."""
        best_url: Optional[str] = None
        best_score = -1
        ent_lower = entity.lower()
        for entry in self._entries(app_id):
            entities = {
                e.split(".")[-1].lower()
                for e in (entry.get("entities", []) or [])
            } | {
                e.split(".")[-1].lower()
                for e in (entry.get("entity_fields", {}) or {}).keys()
            }
            if ent_lower not in entities:
                continue
            url = entry.get("service_url") or ""
            score = _home_score(entity, url)
            if score > best_score:
                best_score = score
                best_url = url or None
        return best_url


def _home_score(entity: str, svc_url: str) -> int:
    entity_base = ts.ent_slug(entity).rstrip("s")
    svc_slug = (svc_url or "").rstrip("/").split("/")[-1].lower()
    if not svc_slug:
        return 0
    if entity_base.startswith(svc_slug) and (
        len(entity_base) == len(svc_slug)
        or (len(entity_base) > len(svc_slug) and entity_base[len(svc_slug)] == "-")
    ):
        return len(svc_slug)
    return 0


def _entity_word_stems(entity: str) -> Set[str]:
    parts = ts.ent_slug(entity).split("-")
    return {ts.stem(p) for p in parts if p and p not in ts.NOISE}


@dataclass
class ResolvedEntity:
    name: Optional[str] = None
    score: float = 0.0
    service_url: Optional[str] = None
    source: Optional[str] = None  # fiori | alias | exact | compound | fuzzy | None


_FUZZY_THRESHOLD = 2.0


class EntityResolver:
    """Stateless resolver. Construct once with an EntityRegistry."""

    def __init__(self, registry: EntityRegistry):
        self._registry = registry

    def resolve(
        self,
        signals: ts.PlanSignals,
        app_id: Optional[str],
        fiori_context: Optional[Dict] = None,
    ) -> ResolvedEntity:
        candidates = self._registry.get_entities(app_id)
        if not candidates:
            return ResolvedEntity()

        by_lower = {c.lower(): c for c in candidates}
        norm = signals.norm

        # 1. Fiori context entity hint (the on-screen record's type).
        hinted = self._fiori_entity_hint(fiori_context)
        if hinted and hinted.lower() in by_lower:
            name = by_lower[hinted.lower()]
            return ResolvedEntity(name, 1.0, self._svc(app_id, name), "fiori")

        # 2. Alias exact match.
        aliases = self._registry.get_aliases(app_id)
        if aliases:
            probe_terms = set(signals.tokens)
            if signals.primary_noun:
                probe_terms.add(signals.primary_noun)
            for term in probe_terms:
                canonical = aliases.get(term)
                if canonical and canonical.lower() in by_lower:
                    name = by_lower[canonical.lower()]
                    return ResolvedEntity(name, 0.95, self._svc(app_id, name), "alias")

        # 3. Primary-noun exact / singular, else full-message containment.
        exact = self._exact_match(by_lower, signals)
        if exact:
            upgraded = self._compound_refine(exact, candidates, norm)
            if upgraded and upgraded != exact:
                return ResolvedEntity(upgraded, 0.9, self._svc(app_id, upgraded), "compound")
            return ResolvedEntity(exact, 1.0, self._svc(app_id, exact), "exact")

        # 5. Fuzzy scoring.
        fuzzy = self._fuzzy_match(candidates, signals)
        if fuzzy:
            return ResolvedEntity(fuzzy, self._last_fuzzy_score, self._svc(app_id, fuzzy), "fuzzy")

        # 6. No-guess.
        return ResolvedEntity()

    # ── internals ────────────────────────────────────────────────────────────

    def _svc(self, app_id: Optional[str], entity: str) -> Optional[str]:
        try:
            return self._registry.service_url_for(app_id, entity)
        except Exception:
            return None

    @staticmethod
    def _fiori_entity_hint(fiori_context: Optional[Dict]) -> Optional[str]:
        if not fiori_context:
            return None
        for key in ("entity", "entityName", "entity_name", "entitySet", "entity_set", "target_entity"):
            v = fiori_context.get(key)
            if isinstance(v, str) and v:
                return v.split(".")[-1]
        ed = fiori_context.get("entity_data") or fiori_context.get("entityData")
        if isinstance(ed, dict):
            meta = ed.get("__metadata")
            if isinstance(meta, dict) and isinstance(meta.get("type"), str):
                return meta["type"].split(".")[-1]
        return None

    @staticmethod
    def _exact_match(by_lower: Dict[str, str], signals: ts.PlanSignals) -> Optional[str]:
        pn = signals.primary_noun
        if pn:
            for cand in (pn, ts.singular(pn)):
                if cand in by_lower:
                    return by_lower[cand]
        # Full-message containment: a candidate name appears as a whole word.
        toks = set(signals.tokens)
        for low, original in by_lower.items():
            if low in toks or ts.singular(low) in toks:
                return original
        return None

    @staticmethod
    def _compound_refine(matched: str, candidates: List[str], norm: str) -> Optional[str]:
        """If `matched` is a prefix of a longer compound entity whose extra
        camelCase parts also appear in the message, upgrade to the compound."""
        m_low = matched.lower()
        best = None
        for cand in candidates:
            cl = cand.lower()
            if cl != m_low and cl.startswith(m_low) and len(cl) > len(m_low):
                extra_parts = ts.ent_slug(cand[len(matched):]).split("-")
                extra_parts = [p for p in extra_parts if p]
                if extra_parts and all(p in norm for p in extra_parts):
                    if best is None or len(cand) > len(best):
                        best = cand
        return best

    def _fuzzy_match(self, candidates: List[str], signals: ts.PlanSignals) -> Optional[str]:
        content_stems = {ts.stem(t) for t in signals.content_tokens}
        content_tokens = signals.content_tokens
        pn_stem = ts.stem(signals.primary_noun) if signals.primary_noun else None

        best_name: Optional[str] = None
        best_score = 0.0
        for cand in candidates:
            ent_words = _entity_word_stems(cand)
            if not ent_words:
                continue
            overlap = len(ent_words & content_stems) / len(ent_words)
            tri = max((ts.trigram_similarity(cand, t) for t in content_tokens), default=0.0)
            boost = 1.0 if (pn_stem and pn_stem in ent_words) else 0.0
            score = overlap * 3.0 + tri * 2.0 + boost * 3.0
            if score > best_score:
                best_score = score
                best_name = cand

        if best_name is not None and best_score >= _FUZZY_THRESHOLD:
            self._last_fuzzy_score = round(min(best_score / 6.0, 0.85), 4)
            return best_name
        return None

    _last_fuzzy_score: float = 0.0
