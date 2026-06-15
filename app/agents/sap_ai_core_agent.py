

import logging
import asyncio
import os
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

import aiohttp
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)


class SAPAICoreAuth:
    """OAuth2 authentication client for SAP AI Core with token caching"""

    def __init__(self, auth_url: str, client_id: str, client_secret: str):
        self.auth_url = auth_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expiry = 0

    async def get_token(self) -> str:
        """Get valid OAuth2 access token (cached and auto-refreshed)"""
        if self.access_token and time.time() < self.token_expiry:
            return self.access_token

        token_endpoint = (
            self.auth_url
            if self.auth_url.endswith("/oauth/token")
            else f"{self.auth_url}/oauth/token"
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                token_endpoint,
                data={"client_id": self.client_id, "client_secret": self.client_secret, "grant_type": "client_credentials"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Authentication failed: {error_text}")

                data = await response.json()
                self.access_token = data.get("access_token")
                expires_in = data.get("expires_in", 3600)
                self.token_expiry = time.time() + expires_in - 60
                return self.access_token


class SAPAICoreAgent:
    """SAP AI Core agent with orchestration service integration"""

    def __init__(self, url: str, auth_url: str, client_id: str, client_secret: str,
                 model_id: str = "gpt-4o", deployment_id: str = "default"):
        self.url = url.rstrip("/")
        self.model_id = model_id
        self.deployment_id = deployment_id
        self.request_count = 0

        self.auth_client = SAPAICoreAuth(auth_url=auth_url, client_id=client_id, client_secret=client_secret)
        self.inference_url = f"{self.url}/v2/inference/deployments/{self.deployment_id}/completion"

        logger.info(f"SAP AI Core Agent configured - model: {model_id}, deployment: {deployment_id}")

    # ── Schema-hint helpers (used by _fetch_real_data) ───────────────────────

    def _parse_entity_section(self, entity: str, schema_hint: str) -> str:
        """Return the raw text block for 'entity' from schema_hint.
        Supports ODataProbe format ('## EntityName entity schema'),
        SchemaExtractor format ('## Entity: EntityName'), and plain ('## EntityName')."""
        import re as _re
        for pat in [
            rf'^##\s+{_re.escape(entity)}\s+entity\s+schema\s*$',
            rf'^##\s+Entity:\s+{_re.escape(entity)}\s*$',
            rf'^##\s+{_re.escape(entity)}\s*$',
        ]:
            m = _re.search(pat + r'(.*?)(?=^##\s|\Z)',
                           schema_hint, _re.MULTILINE | _re.DOTALL)
            if m:
                return m.group(1)
        return ""

    def _parse_entity_fields(self, entity: str, schema_hint: str) -> list:
        """Extract scalar field names for 'entity' from schema_hint.
        Handles ODataProbe inline format ('Fields: f1 (Type), f2 (Type)')
        and SchemaExtractor bullet format ('- fieldName (Type...)')."""
        import re as _re
        section = self._parse_entity_section(entity, schema_hint)
        if not section:
            return []
        fields: list = []
        for line in section.splitlines():
            m = _re.match(r'^(?:Fields|Key\s+fields|Key fields):\s+(.+)$', line.strip())
            if m:
                for item in m.group(1).split(','):
                    fn = _re.match(r'^\s*(\w+)\s*\(', item)
                    if fn and fn.group(1) not in fields:
                        fields.append(fn.group(1))
        for match in _re.finditer(r'^-\s+(\w+)\s*\(', section, _re.MULTILINE):
            name = match.group(1)
            if name not in fields:
                fields.append(name)
        return fields

    def _parse_associations(self, entity: str, schema_hint: str) -> list:
        """Return list of (navPropName, targetEntityName) for 'entity'.
        Handles ODataProbe ('Navigation: nav → Target[]') and
        SchemaExtractor ('- navProp → Target (association...)') formats."""
        import re as _re
        section = self._parse_entity_section(entity, schema_hint)
        if not section:
            return []
        assocs: list = []
        for line in section.splitlines():
            m = _re.match(r'^Navigation:\s+(.+)$', line.strip())
            if m:
                for item in m.group(1).split(','):
                    pair = _re.match(r'^\s*(\w+)\s*(?:→|->|>)\s*(\w+)', item.strip())
                    if pair:
                        nav, tgt = pair.group(1), pair.group(2).rstrip('[]')
                        if (nav, tgt) not in assocs:
                            assocs.append((nav, tgt))
        for match in _re.finditer(r'^-\s+(\w+)\s+(?:→|->|>)\s+(\w+)\s*\(', section, _re.MULTILINE):
            nav, tgt = match.group(1), match.group(2)
            if (nav, tgt) not in assocs:
                assocs.append((nav, tgt))
        return assocs

    def _build_fk_filter(self, message: str, fields: list) -> Optional[str]:
        """Match a bare number in the message to the most semantically appropriate FK field.

        Handles cases like 'materials for blend 2466' where 'blend' in message
        matches 'Blend' in the FK field 'to_FertilizerBlend_orderID'.
        """
        import re as _re
        numbers = _re.findall(r'\b(\d{3,})\b', message)
        if not numbers:
            return None
        fk_fields = [
            f for f in fields
            if _re.search(r'(?i)(orderID|order_id|_ID|ID|Number)$', f)
            and f.lower() not in ('id', 'areatype_id', 'crop_id', 'variety_id')
        ]
        if not fk_fields:
            return None
        msg_lower = message.lower()
        msg_words = set(_re.split(r'\W+', msg_lower))
        best_fk = None
        best_score = 0
        for fk in fk_fields:
            parts = [p.lower() for p in _re.split(r'(?<=[a-z])(?=[A-Z])|_|(?<=[A-Z])(?=[A-Z][a-z])', fk) if len(p) > 2]
            score = sum(
                1 for p in parts
                if p in msg_words
                or any(w.startswith(p[:4]) for w in msg_words if len(w) >= 4 and len(p) >= 4)
            )
            if score > best_score:
                best_score = score
                best_fk = fk
        if best_fk and best_score > 0:
            return f"{best_fk} eq {numbers[0]}"
        return None

    def _parse_fields_from_rag(self, entity: str, rag_context: str) -> list:
        """Extract field names for a specific entity from RAG context text."""
        import re as _re
        if not rag_context:
            return []
        m = _re.search(
            rf'Entity:\s+{_re.escape(entity)}.*?Available fields:\s+([^\n]+)',
            rag_context, _re.DOTALL | _re.IGNORECASE,
        )
        if m:
            return [f.strip() for f in m.group(1).split(',') if f.strip()]
        section_m = _re.search(
            rf'(?:Entity:\s+|## ){_re.escape(entity)}(.*?)(?=Entity:\s+|^## |\Z)',
            rag_context, _re.DOTALL | _re.IGNORECASE | _re.MULTILINE,
        )
        if section_m:
            return _re.findall(r'^\s*-\s+(\w+)\s*\(', section_m.group(1), _re.MULTILINE)
        return []

    def _build_filter(self, message: str, fields: list) -> Optional[str]:
        """Build an OData $filter clause from natural language using known field names.

        Patterns detected:
          - '{field} is/= {value}' or 'with {field} {value}'  → field eq 'value'
          - '{field} greater/less than {n}'                    → field gt/lt n
          - Bare ALL_CAPS word (e.g. PENDING) near a status field → status eq 'PENDING'
        """
        import re as _re
        COMPARISON_WORDS = {'greater', 'less', 'more', 'above', 'below', 'than',
                            'at', 'most', 'least', 'equal', 'between'}
        filters: list = []
        for field in fields:
            fl = _re.escape(field.lower())
            matched = False
            # 1. Numeric comparisons FIRST — prevents 'greater' being captured as a value
            for op_re, odata_op in [
                (r'(?:greater\s+than|more\s+than|above|>)\s+', 'gt'),
                (r'(?:less\s+than|below|<)\s+',               'lt'),
                (r'(?:at\s+least|>=)\s+',                     'ge'),
                (r'(?:at\s+most|<=)\s+',                      'le'),
            ]:
                hit = _re.search(rf'\b{fl}\s+(?:is\s+|are\s+)?{op_re}(\d+(?:\.\d+)?)', message, _re.IGNORECASE)
                if hit:
                    filters.append(f"{field} {odata_op} {hit.group(1)}")
                    matched = True
                    break
            if matched:
                continue
            # 2. Explicit eq patterns — search original message to preserve value case
            for pat in [
                rf'\b{fl}\s+(?:is|eq|=|:)\s+["\']?([A-Za-z0-9_\-]+)["\']?',
                rf'\bwith\s+{fl}\s+(?:of\s+)?["\']?([A-Za-z0-9_\-]+)["\']?',
                rf'\b{fl}\s+["\']([^"\']+)["\']',
            ]:
                hit = _re.search(pat, message, _re.IGNORECASE)
                if hit:
                    val = hit.group(1)
                    # Ignore if the captured token is itself a comparison keyword
                    if val.lower() in COMPARISON_WORDS:
                        continue
                    filters.append(
                        f"{field} eq {val}"
                        if _re.match(r'^\d+(\.\d+)?$', val)
                        else f"{field} eq '{val}'"
                    )
                    matched = True
                    break
        # Fallback: bare ALL_CAPS token (e.g. PENDING, APPROVED, DRAFT)
        # → apply to first status-like field when nothing else matched
        if not filters:
            enum_hit = _re.search(r'\b([A-Z_]{3,})\b', message)
            if enum_hit:
                status_fields = [
                    f for f in fields
                    if f.lower() in ('status', 'state', 'type', 'category',
                                     'priority', 'phase', 'stage')
                ]
                if status_fields:
                    filters.append(f"{status_fields[0]} eq '{enum_hit.group(1)}'")
        return " and ".join(filters) if filters else None

    def _build_expand(self, message: str, assocs: list) -> Optional[str]:
        """Build OData $expand when the message mentions terms matching
        a navigation property name or its target entity (camelCase-aware)."""
        import re as _re
        msg_lower = message.lower()
        expand: list = []
        for nav_prop, target in assocs:
            tokens: set = {nav_prop.lower(), target.lower()}
            tokens |= {w.lower() for w in _re.split(r'(?<=[a-z])(?=[A-Z])|_', nav_prop) if len(w) > 2}
            tokens |= {w.lower() for w in _re.split(r'(?<=[a-z])(?=[A-Z])|_', target)   if len(w) > 2}
            if any(_re.search(r'\b' + _re.escape(t) + r'\b', msg_lower) for t in tokens):
                expand.append(nav_prop)
        return ",".join(expand) if expand else None

    def _aggregate_python(self, rows: list, group_field: str) -> str:
        """Group rows by a field in Python and return a breakdown string.
        Field matching is case-insensitive so 'status' matches 'Status'."""
        counts: dict = {}
        resolved: Optional[str] = None
        for row in rows:
            if resolved is None:
                for k in row.keys():
                    if k.lower() == group_field.lower():
                        resolved = k
                        break
            key = str(row.get(resolved, "unknown")) if resolved else "unknown"
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return f"  No rows returned for grouping by '{group_field}'."
        label = resolved or group_field
        lines = [f"  Count by {label}:"]
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    # ── Main data fetcher ─────────────────────────────────────────────────────

    async def _fetch_real_data(
        self,
        fiori_context: Optional[Dict[str, Any]],
        odata_token: Optional[str],
        message: str,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        rag_context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """
        Detect data-retrieval questions and fetch live data from the app's OData service.
        Returns a formatted block prepended to the user message so the LLM answers
        with real data instead of generic guidance.

        Three enhancements — all automatic, no app-side config required:
          1. NL → $filter   "show PENDING blends"   → $filter=status eq 'PENDING'
          2. Auto $expand   "blends with materials"  → $expand=materials
          3. Python groupby "blends by status"       → group fetched rows in Python
        """
        service_url = None
        schema_hint_override = ""

        if fiori_context:
            service_url = (
                fiori_context.get("service_url") or fiori_context.get("serviceUrl")
            )
            logger.debug("[fetch_real_data] fiori_context present, service_url=%s", service_url)
        else:
            logger.debug("[fetch_real_data] no fiori_context, will try registry")

        # ── Registry fallback ─────────────────────────────────────────────────
        # If the widget did NOT forward a service_url (standalone chat, no widget
        # embedded), fall back to the first registered service for this app_id.
        # This allows live OData queries even when fiori_context is empty.
        _registry_base_url = ""  # CAP server base URL stored at registration time
        if not service_url and app_id:
            try:
                from app.api.apps import _service_tool_registry
                services = _service_tool_registry.get(app_id, [])
                if services:
                    service_url = services[0].get("service_url", "")
                    _registry_base_url = services[0].get("app_base_url", "")
                    logger.info("[fetch_real_data] no widget service_url — fell back to registry: %s", service_url)
            except Exception:
                pass
        elif app_id:
            # Fetch app_base_url even when we already have a service_url from the widget
            try:
                from app.api.apps import _service_tool_registry
                services = _service_tool_registry.get(app_id, [])
                if services:
                    _registry_base_url = services[0].get("app_base_url", "")
            except Exception:
                pass

        if not service_url:
            logger.info("[fetch_real_data] no service_url found for app_id=%s — skipping live fetch", app_id)
            return None

        import re as _re

        # ── Context extension for very short follow-up messages ───────────────
        # "for 2466?" after asking about materials → extend with last user message
        # so entity matching can use the topic from the prior turn.
        if len(message.strip()) < 35 and history:
            last_user = next(
                (h.get('content', '') for h in reversed(history) if h.get('role') == 'user'),
                None
            )
            if last_user and last_user.strip() != message.strip():
                message = f"{message.strip()} {last_user[:200]}"
                logger.info("[fetch_real_data] short message extended with history context")

        DATA_QUESTION = _re.compile(
            r"\b("
            r"how many|count|total"
            r"|list|show|display|give me|tell me"
            r"|get|fetch|find|search|look"
            r"|what(?:\s+are|\s+is)?\s+(?:the|all|available)?"
            r"|available|exist|present"
            r"|all|any|records?|entries|data|items?"
            r")",
            _re.IGNORECASE,
        )
        if not DATA_QUESTION.search(message):
            return None

        # ── Extract the PRIMARY NOUN — the main subject of the question ───────
        # "what are the MATERIALS for..." → primary = "materials"
        # This is used to boost matching entities whose name contains the subject,
        # preventing secondary context words ("sales order") from winning.
        _pn_match = _re.search(
            r'\b(?:what\s+(?:are\s+)?(?:the\s+)?|show\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?'
            r'|list\s+(?:all\s+)?(?:the\s+)?|get\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?'
            r'|fetch\s+(?:all\s+)?(?:the\s+)?|find\s+(?:all\s+)?(?:the\s+)?)\s*(\w+)',
            message, _re.IGNORECASE,
        )
        _primary_noun = _pn_match.group(1).lower() if _pn_match else None

        schema_hint = (
            (fiori_context.get("extra") or {}).get("schema_hint") or ""
            if fiori_context else ""
        )
        # Match headings like "## FertilizerBlend entity schema" OR "## FertilizerBlend"
        raw_entities: list = _re.findall(
            r"^###?\s+([A-Za-z][A-Za-z0-9_]+)(?:\s+entity\s+schema)?\s*$",
            schema_hint, _re.MULTILINE
        )
        entity_names = [
            e for e in raw_entities
            if not e.lower().startswith(("service", "available"))
        ]
        # Fallback: parse "Entities: Foo, Bar, Baz" line from the summary section
        if not entity_names:
            m = _re.search(r'^Entities:\s+(.+)$', schema_hint, _re.MULTILINE)
            if m:
                entity_names = [e.strip() for e in m.group(1).split(",") if e.strip()]

        # ── Cross-service entity pool ─────────────────────────────────────────
        # The widget schema_hint only covers the service currently in view.
        # Build map:  entity_name → service_url  from the service tool registry,
        # preferring each entity's "home" service.
        #
        # Home-service score: length of the longest service-URL slug that is a
        # clean PREFIX of the entity slug (handles plural set-names and V1 suffixes).
        # e.g. "SalesOrders" → entity_base "sales-order" starts with service slug
        #      "sales-order" (len=11) → score 11, beats "fertilizer-blend" (score 0).
        def _ent_slug(entity: str) -> str:
            return _re.sub(r'(?<=[a-z])(?=[A-Z])', '-', entity).lower()

        def _home_score(entity: str, svc_url: str) -> int:
            entity_base = _ent_slug(entity).rstrip('s')  # normalise plural
            svc_slug = svc_url.rstrip('/').split('/')[-1].lower()
            if not svc_slug:
                return 0
            if entity_base.startswith(svc_slug) and (
                len(entity_base) == len(svc_slug)
                or (len(entity_base) > len(svc_slug) and entity_base[len(svc_slug)] == '-')
            ):
                return len(svc_slug)  # longer match = more specific
            return 0

        _entity_to_service: Dict[str, str] = {}
        _entity_score: Dict[str, int] = {}      # home-score for current mapping
        if app_id:
            try:
                from app.api.apps import _service_tool_registry
                services = _service_tool_registry.get(app_id, [])
                for svc in services:
                    reg_svc_url = svc.get("service_url", "")
                    for ent in svc.get("entities", []):
                        score = _home_score(ent, reg_svc_url)
                        # Overwrite if this service is a better "home" for the entity
                        if score > _entity_score.get(ent, -1):
                            _entity_to_service[ent] = reg_svc_url
                            _entity_score[ent] = score
            except Exception as e:
                logger.warning("[fetch_real_data] registry lookup failed: %s", e)

        # Track the original schema_hint service so we can detect cross-service
        _schema_hint_service = service_url

        # Merge schema_hint entities (current service default) + registry entities
        all_candidate_entities: list = list(
            dict.fromkeys(entity_names + list(_entity_to_service.keys()))
        )
        if not all_candidate_entities:
            return None

        # ── Trigram helper for typo-tolerant matching ─────────────────────────
        def _trigrams(s: str) -> set:
            s = _re.sub(r'\W+', '', s.lower())
            return {s[i:i+3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

        # ── Entity matching — scored across ALL services ───────────────────────
        # Generic English filler/question words that are never entity names.
        # DO NOT put app-specific terminology here — derive it dynamically below.
        _NOISE = {
            "are", "be", "been", "can", "could", "did", "do", "does", "get",
            "give", "has", "have", "how", "in", "is", "it", "its", "list",
            "make", "many", "me", "of", "our", "show", "tell", "that", "the",
            "them", "there", "these", "this", "those", "to", "total", "us",
            "was", "were", "what", "which", "who", "will", "with", "would",
            "all", "any", "each", "fetch", "find", "from", "give", "their",
        }

        # Dynamically extend noise with words from the app name / app_id so that
        # e.g. "blending" in "stutsman-blending" or "procurement" in a purchasing
        # app don't accidentally score against entity names.
        if app_id:
            for raw in _re.split(r'[\s\-_]+', app_id):
                w = raw.lower()
                if len(w) > 2:
                    _NOISE.add(w)
        # Also extract words from app_name stored in the registry
        if app_id:
            try:
                from app.api.apps import _service_tool_registry
                _svcs = _service_tool_registry.get(app_id, [])
                if _svcs:
                    _app_name = _svcs[0].get("app_name", "")
                    for raw in _re.split(r'[\s\-_]+', _app_name):
                        w = raw.lower()
                        if len(w) > 2:
                            _NOISE.add(w)
            except Exception:
                pass
        msg_lower = message.lower()
        msg_words = {w for w in _re.split(r'\W+', msg_lower) if w and w not in _NOISE}
        # Compute trigrams on the full cleaned message for typo tolerance
        msg_tris = _trigrams(_re.sub(r'\W+', '', msg_lower))

        best_entity: Optional[str] = None
        best_service_url: str = service_url  # default to current widget service

        # Pass 1: exact containment (highest confidence)
        for ent in all_candidate_entities:
            if ent.lower() in msg_lower:
                best_entity = ent
                best_service_url = _entity_to_service.get(ent, service_url)
                break

        # Pass 2: combined word-stem overlap + trigram similarity
        # word_overlap handles normal spelling; trigrams handle typos.
        if not best_entity:
            scored = []
            for ent in all_candidate_entities:
                ent_words = set(
                    w.lower()
                    for w in _re.split(r'(?<=[a-z])(?=[A-Z])|_', ent)
                    if len(w) > 2
                )
                stems = set()
                for w in ent_words:
                    stems.add(w)
                    if w.endswith("tions"): stems.add(w[:-5])
                    if w.endswith("tion"):  stems.add(w[:-4])
                    if w.endswith("es"):    stems.add(w[:-2])
                    if w.endswith("s"):     stems.add(w[:-1])
                word_overlap = sum(
                    1 for mw in msg_words
                    if any(mw.startswith(stem[:4]) or stem.startswith(mw[:4])
                           for stem in stems if len(stem) >= 4)
                )
                # Trigram similarity: handles typos like "slaesorders" → SalesOrder
                ent_tris = _trigrams(ent)
                tri_sim = len(msg_tris & ent_tris) / max(len(msg_tris | ent_tris), 1)

                # Primary-noun boost: strongly prefer entity whose name directly
                # contains the main SUBJECT of the question.
                # Example: "materials for blend 2466" → primary="materials" →
                # AssignMaterialToBlend gets +20 over SalesOrders even though
                # "sales order" appears in the message context.
                primary_boost = 0.0
                if _primary_noun and len(_primary_noun) >= 4:
                    pn4 = _primary_noun[:4]
                    if any(s.startswith(pn4) or pn4.startswith(s[:4]) for s in stems if len(s) >= 4):
                        primary_boost = 20.0

                score = word_overlap * 3.0 + tri_sim * 10.0 + primary_boost
                scored.append((score, ent))
            scored.sort(reverse=True)
            logger.info(
                "[fetch_real_data] top-5 scores: %s",
                [(round(s, 2), e) for s, e in scored[:5]]
            )
            if scored and scored[0][0] > 0:
                best_entity = scored[0][1]
                best_service_url = _entity_to_service.get(best_entity, service_url)

        # Pass 3: if nothing scored > 0 across all candidates, don't guess.
        # Return None so the AI uses RAG context instead of fetching wrong data.
        if not best_entity:
            logger.info(
                "[fetch_real_data] no entity matched (score=0 for all %d candidates) — skipping live fetch",
                len(all_candidate_entities)
            )
            return None

        # Use whichever service URL the winning entity belongs to
        service_url = best_service_url
        _cross_service = (
            service_url.rstrip("/") != _schema_hint_service.rstrip("/")
        )

        logger.info(
            "[fetch_real_data] resolved entity=%s from %s (cross_service=%s)",
            best_entity, service_url, _cross_service
        )

        # ── Build OData query options from NL ─────────────────────────────────
        # For cross-service entities, schema_hint doesn't cover their fields.
        # Fall back to RAG context (which contains schema summaries for ALL entities).
        if not _cross_service:
            fields = self._parse_entity_fields(best_entity, schema_hint)
            assocs = self._parse_associations(best_entity, schema_hint)
        else:
            fields = self._parse_fields_from_rag(best_entity, rag_context or "")
            assocs = []

        odata_filter = self._build_filter(message, fields) if fields else None
        if not odata_filter:
            # FK filter: try to match a bare number in the message to a FK field
            odata_filter = self._build_fk_filter(message, fields) if fields else None
        odata_expand = self._build_expand(message, assocs) if assocs else None

        # ── User-scoped filter: "my ...", "I have ...", "assigned to me" ───────
        # When the user mentions ownership language AND we know who they are,
        # add a createdBy/owner filter so the AI sees only their own records.
        # CAP managed entities always populate `createdBy` with the user_name from JWT.
        _MY_INTENT = _re.compile(
            r"\b(my|mine|i\s+have|i\s+created|i\s+made|assigned\s+to\s+me|my\s+own)\b",
            _re.IGNORECASE,
        )
        if user_id and _MY_INTENT.search(message) and fields:
            # Prefer `createdBy` (CAP managed), then generic owner synonyms
            owner_field = next(
                (f for f in fields if f.lower() in ("createdby", "owner", "assignedto", "userid", "createdbyuser")),
                None,
            )
            if owner_field:
                user_filter = f"{owner_field} eq '{user_id}'"
                odata_filter = f"{odata_filter} and {user_filter}" if odata_filter else user_filter

        # Aggregation intent: "by status", "per category", "grouped by type"
        AGG_PAT = _re.compile(
            r'\b(?:by|per|group(?:ed)?\s+by|breakdown\s+by)\s+(\w+)',
            _re.IGNORECASE
        )
        agg_match = AGG_PAT.search(message)
        group_field: Optional[str] = agg_match.group(1) if agg_match else None

        # ── Resolve relative URL ──────────────────────────────────────────────
        # Use the app_base_url stored at registration time (e.g. http://localhost:4004).
        # Falls back to localhost:4004 only when the SDK pre-dates app_base_url support.
        if service_url.startswith("/"):
            base = _registry_base_url.rstrip("/") if _registry_base_url else "http://localhost:4004"
            service_url = f"{base}{service_url}"

        logger.info(
            "[fetch_real_data] entity=%s service=%s filter=%s expand=%s",
            best_entity, service_url, odata_filter, odata_expand
        )

        try:
            import aiohttp as _aio
            import json as _json

            headers: dict = {"Accept": "application/json"}
            if odata_token:
                headers["Authorization"] = (
                    f"Bearer {odata_token.replace('Bearer ', '').replace('bearer ', '')}"
                )

            base_url = f"{service_url.rstrip('/')}/{best_entity}"
            count_val: Optional[int] = None
            rows_data = None

            async with _aio.ClientSession() as session:
                # Aggregation needs a larger sample; otherwise fetch up to 20 rows
                top = 200 if group_field else 20
                list_params: dict = {"$top": top}
                if odata_filter:
                    list_params["$filter"] = odata_filter
                if odata_expand:
                    list_params["$expand"] = odata_expand
                if not group_field:
                    list_params["$count"] = "true"

                # Fetch exact count (skip for aggregation — the grouping is the count)
                if not group_field:
                    count_params: dict = {}
                    if odata_filter:
                        count_params["$filter"] = odata_filter
                    async with session.get(
                        base_url + "/$count", params=count_params,
                        headers=headers, timeout=_aio.ClientTimeout(total=8),
                    ) as resp:
                        if resp.status == 200:
                            try:
                                count_val = int((await resp.text()).strip())
                            except ValueError:
                                pass

                # Fetch rows (with optional filter + expand)
                async with session.get(
                    base_url, params=list_params,
                    headers=headers, timeout=_aio.ClientTimeout(total=8),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        rows_data = data.get("value", [])
                        if count_val is None and not group_field:
                            total = data.get("@odata.count")
                            if total is not None:
                                count_val = int(total)

            if count_val is None and rows_data is None:
                return None

            # ── Format output for the LLM ─────────────────────────────────────
            lines = [f"[Real data from OData — {best_entity}]"]
            applied = []
            if odata_filter:
                applied.append(f"$filter={odata_filter}")
            if odata_expand:
                applied.append(f"$expand={odata_expand}")
            if group_field:
                applied.append(f"grouped by '{group_field}'")
            if applied:
                lines.append(f"Applied: {', '.join(applied)}")

            if group_field and rows_data is not None:
                lines.append(f"Fetched {len(rows_data)} record(s) for grouping.")
                lines.append(self._aggregate_python(rows_data, group_field))
            else:
                if count_val is not None:
                    lines.append(f"Total record count: {count_val}")
                if rows_data:
                    sample = rows_data[:5]
                    lines.append(f"Sample records ({len(sample)} of {count_val or len(rows_data)}):")

                    # Collect scalar fields, skip UUIDs unless they're the only key
                    uuid_re = _re.compile(
                        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                        _re.IGNORECASE
                    )
                    all_keys: list = []
                    for row in sample:
                        for k in row:
                            if k not in all_keys:
                                all_keys.append(k)

                    # Prefer non-UUID columns; fall back to all if everything is UUID
                    scalar_keys = [
                        k for k in all_keys
                        if not isinstance(sample[0].get(k), (dict, list))
                        and sample[0].get(k) is not None
                    ]
                    meaningful_keys = [
                        k for k in scalar_keys
                        if not (
                            isinstance(sample[0].get(k), str)
                            and uuid_re.match(str(sample[0].get(k, "")))
                            and k.lower() not in ("id", "key")
                        )
                    ] or scalar_keys

                    # Emit as a simple CSV-style table the LLM can use to build Markdown
                    if meaningful_keys:
                        lines.append("| " + " | ".join(meaningful_keys) + " |")
                        lines.append("|" + "|".join("---" for _ in meaningful_keys) + "|")
                        for row in sample:
                            cells = [str(row.get(k, "")) for k in meaningful_keys]
                            lines.append("| " + " | ".join(cells) + " |")
                    else:
                        for i, row in enumerate(sample, 1):
                            lines.append(f"  {i}. {_json.dumps(row, default=str)}")
            lines.append("[End of live data]\n")
            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"OData fetch skipped ({best_entity}): {e}")
            return None

    async def _fetch_rag_context(self, message: str, app_id: Optional[str]) -> Optional[str]:
        """Retrieve relevant chunks from the vector store for the given query + app_id."""
        if not app_id:
            return None
        try:
            from app.knowledge.knowledge_base import get_knowledge_base
            kb = get_knowledge_base()
            ctx = kb.search_with_app_context(query=message, app_id=app_id)
            return ctx if ctx else None
        except Exception as e:
            logger.warning(f"RAG context fetch failed for app '{app_id}': {e}")
            return None

    def _build_system_message(
        self,
        rag_context: Optional[str] = None,
        app_id: Optional[str] = None,
        fiori_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        import re as _re

        # ── Extract dynamic app metadata from fiori_context ──────────────────
        app_name = app_id
        schema_hint = None
        service_url = None

        if fiori_context and isinstance(fiori_context, dict):
            app_name = (
                fiori_context.get("app_name")
                or fiori_context.get("appName")
                or fiori_context.get("app_id")
                or fiori_context.get("appId")
                or app_id
            )
            service_url = fiori_context.get("service_url") or fiori_context.get("serviceUrl")
            extra = fiori_context.get("extra") or {}
            if isinstance(extra, dict):
                schema_hint = extra.get("schema_hint")

        # ── Optionally load stored raw $metadata XML for deeper schema context ─
        raw_xml_section = ""
        if app_id and service_url:
            try:
                from app.api.apps import load_metadata_xml
                raw_xml = load_metadata_xml(app_id, service_url)
                if raw_xml and len(raw_xml) < 300_000:  # safety cap: skip absurdly large XMLs
                    raw_xml_section = (
                        "\n\n[Raw $metadata XML — use only for deep schema inspection]\n"
                        f"```xml\n{raw_xml[:50_000]}\n```"  # cap at 50 KB to stay within context window
                    )
            except Exception:
                pass

        # ── Parse entity names from schema_hint (lines starting with ##) ─────
        entity_names: list[str] = []
        if schema_hint:
            entity_names = [
                m.strip()
                for m in _re.findall(r'^##\s+(.+?)\s*$', schema_hint, _re.MULTILINE)
                if m.strip()
            ]

        app_label = f'"{app_name}"' if app_name else "this SAP application"

        # ── Dynamic entity section injected into system prompt ────────────────
        entity_section = ""
        if entity_names:
            entity_list = ", ".join(f"`{e}`" for e in entity_names[:25])
            entity_section = (
                f"\n\nCURRENT APP ENTITIES — {app_label} exposes these OData entities: "
                f"{entity_list}.\n"
                "When the user mentions ANY business term (records, items, entries, incidents, "
                "orders, jobs, tasks, blends, formulas, materials, components, documents, or any "
                "domain-specific word), identify which of the entities above is the closest "
                "semantic match and answer in terms of that entity. "
                "Do NOT require the user to use the exact technical entity name."
            )

        base = (
            "You are BTP Copilot, an intelligent AI assistant that can be embedded in "
            "ANY SAP Fiori / CAP application. Each session belongs to a DIFFERENT application "
            "with its own unique entities, terminology and OData services.\n\n"

            "ABSOLUTE RULES — follow every one without exception:\n\n"

            "1. SCHEMA IS GROUND TRUTH\n"
            "   The user's message begins with '[Context from the Fiori application you are "
            "embedded in]'. This block contains the OData entity schema for the current app. "
            "It is the ONLY authoritative source for what entities and fields exist. "
            "Never rely on prior training knowledge about what an SAP app 'should' have.\n\n"

            "2. NATURAL-LANGUAGE → ENTITY MAPPING\n"
            "   Users speak in business/domain language, never in technical entity names. "
            "Your job is semantic mapping:\n"
            "   • Main transactional objects (blend, order, job, incident, case, ticket, "
            "record, entry, document) → look for the primary entity in the schema\n"
            "   • Detail / line items (ingredient, component, material, line, item, part) "
            "→ look for child / composition entities\n"
            "   • Reference data (status, priority, type, category, unit) "
            "→ look for code-list / value-help entities\n"
            "   • Match by meaning — spelling similarity is irrelevant.\n\n"

            "3. NEVER DENY EXISTENCE\n"
            "   If any entity in the schema could plausibly match what the user is asking about, "
            "use it. Only say an entity doesn't exist if the schema is present AND you have "
            "checked every entity in it and found no reasonable match.\n\n"

            "3b. NEVER FABRICATE DATA\n"
            "   If no '[Real data from OData]' block is present in the message, do NOT invent "
            "example rows, fake record IDs, fake field values, or placeholder tables. "
            "This is the MOST IMPORTANT rule — making up data destroys user trust. "
            "If the live data block is absent, say: "
            "'I wasn't able to retrieve the live data for this. Please try again in a moment.' "
            "Do NOT give the user raw OData query URLs — users are business users, not developers.\n\n"

            "4. COUNTS & LIVE DATA\n"
            "   The user's message may be prefixed with a '[Real data from OData — EntityName]' "
            "block. When that block is present:\n"
            "   a) Use the exact numbers from it — do NOT say you cannot query the database.\n"
            "   b) State the total count naturally (e.g. 'There are 4 fertilizer blends…').\n"
            "   c) Present the sample records as a **Markdown table** with the most meaningful\n"
            "      scalar fields as columns (skip internal IDs/UUIDs unless they are the only\n"
            "      identifier). Example column choice: Status, Name, Date, Amount, etc.\n"
            "   d) After the table, add 1-2 sentences of brief insight if helpful\n"
            "      (e.g. breakdown by status, date range, notable values).\n"
            "   e) If the live data block is absent, say the data is temporarily unavailable "
            "and ask the user to try again.\n\n"

            "5. STAY IN APP CONTEXT\n"
            "   Every answer must be grounded in the schema provided. "
            "Do not import entity names or field names from previous conversations or "
            "general SAP knowledge — the current app may have completely different entities."
            + entity_section
            + (
                f"\n\n6. CURRENT USER\n"
                f"   The authenticated user is: {user_id}.\n"
                "   When the user says 'my', 'mine', 'I created' or similar, this refers to "
                f"records where the owner/createdBy field equals '{user_id}'.\n"
                "   The live-data block above is already filtered to this user's records when ownership intent was detected."
                if user_id else ""
            )
            + "\n\n"

            "RESPONSE FORMAT RULES (always apply):\n"
            "  • Lead with the direct answer — never start with 'Sure!' / 'Great!' / 'Of course!'.\n"
            "  • Use **Markdown tables** for any set of records (>1 row).\n"
            "  • Use numbered lists for ordered steps; bullet points for unordered options.\n"
            "  • Use **bold** for entity/field names and key values.\n"
            "  • Keep responses concise. Expand only if the user asked for detail.\n"
            "  • When you show a count, follow with a one-line summary (e.g. breakdown by status).\n"
            "  • Never repeat the user's question back to them.\n"
            "  • Do not add disclaimers like 'I don't have access to real-time data' when a "
            "[Real data from OData] block is present — you clearly DO have real data."
        )

        if rag_context:
            base += (
                f"\n\nAdditional retrieved knowledge for {app_label}:\n\n{rag_context}"
            )

        if raw_xml_section:
            base += raw_xml_section

        return base

    async def get_response(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        app_id: Optional[str] = None,
        fiori_context: Optional[Dict[str, Any]] = None,
        odata_token: Optional[str] = None,
        user_id: Optional[str] = None,
        raw_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.request_count += 1
        start_time = datetime.utcnow()

        rag_context = await self._fetch_rag_context(message, app_id)
        system_message = self._build_system_message(rag_context, app_id, fiori_context, user_id)

        # Fetch real live data from the app's OData service when the user
        # is asking a data question (counts, lists, records).
        # Use raw_message (stripped of the fiori context prefix) for entity
        # matching so that e.g. "View: FertilizerBlend" in the prefix does
        # not fool Pass 1 into picking FertilizerBlend for every query.
        _match_msg = raw_message if raw_message else message
        live_data_block = await self._fetch_real_data(
            fiori_context, odata_token, _match_msg,
            user_id, app_id,
            rag_context=rag_context,
            history=history,
        )
        if live_data_block:
            message = live_data_block + message

        token = await self.auth_client.get_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "AI-Resource-Group": "default",
        }

        # Build orchestration template messages
        template_messages = [
            {"role": "system", "content": system_message},
        ]
        if history:
            for msg in history[-10:]:
                template_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        template_messages.append({"role": "user", "content": "{{?user_query}}"})

        payload = {
            "orchestration_config": {
                "module_configurations": {
                    "llm_module_config": {
                        "model_name": self.model_id,
                        "model_params": {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.9}
                    },
                    "templating_module_config": {"template": template_messages}
                }
            },
            "input_params": {"user_query": message}
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.inference_url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                response_text = await response.text()

                if response.status != 200:
                    raise Exception(f"API error {response.status}: {response_text}")

                import json as _json
                try:
                    result = _json.loads(response_text)
                except Exception:
                    raise Exception(f"Failed to parse API response: {response_text[:500]}")

                # Parse orchestration response — log structure to diagnose extraction path
                logger.info(f"API response keys: {list(result.keys())}")

                content = ""
                # Path 1: module_results.llm.choices (orchestration v1)
                module_results = result.get("module_results", {})
                llm_result = module_results.get("llm", {})
                if "choices" in llm_result and len(llm_result["choices"]) > 0:
                    content = llm_result["choices"][0].get("message", {}).get("content", "")
                    logger.info(f"Extracted via module_results.llm.choices ({len(content)} chars)")
                # Path 2: orchestration_result.choices (orchestration v2)
                elif "orchestration_result" in result:
                    orch = result["orchestration_result"]
                    choices = orch.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        logger.info(f"Extracted via orchestration_result.choices ({len(content)} chars)")
                # Path 3: top-level choices
                elif "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    logger.info(f"Extracted via top-level choices ({len(content)} chars)")
                # Path 4: fallback scalar fields
                else:
                    content = result.get("completion") or result.get("text") or result.get("output") or ""
                    logger.warning(f"Fell back to scalar extraction, result keys: {list(result.keys())}")
                    if not content:
                        # Last resort: dump so we can see the structure
                        logger.error(f"Could not extract content. Full response: {response_text[:1000]}")

                response_time = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"Response received ({response_time:.2f}s)")

                return {"response": content, "model": self.model_id, "response_time": response_time}

    async def stream_response(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        app_id: Optional[str] = None,
        fiori_context: Optional[Dict[str, Any]] = None,
        odata_token: Optional[str] = None,
        user_id: Optional[str] = None,
        raw_message: Optional[str] = None,
    ):
        """Stream response token-by-token using word-level chunking."""
        result = await self.get_response(
            message=message,
            history=history,
            app_id=app_id,
            fiori_context=fiori_context,
            odata_token=odata_token,
            user_id=user_id,
            raw_message=raw_message,
        )
        text = result.get("response", "")
        words = text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            if chunk:
                yield chunk
            await asyncio.sleep(0)

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_type": "sap_ai_core",
            "status": "healthy",
            "model": self.model_id,
            "deployment": self.deployment_id,
            "total_requests": self.request_count,
            "api_endpoint": self.inference_url,
        }
