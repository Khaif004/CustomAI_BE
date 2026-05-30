

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
        if not fiori_context:
            return None

        service_url = (
            fiori_context.get("service_url") or fiori_context.get("serviceUrl")
        )
        if not service_url:
            return None

        import re as _re
        DATA_QUESTION = _re.compile(
            r"\b(how many|count|total|list|show|get|fetch|all|any|exist|records?|entries|data)"
            r".*\b",
            _re.IGNORECASE,
        )
        if not DATA_QUESTION.search(message):
            return None

        schema_hint = (
            (fiori_context.get("extra") or {}).get("schema_hint") or ""
        )
        raw_entities: list = _re.findall(
            r"^###?\s+([A-Za-z][A-Za-z0-9_]+)\s*$", schema_hint, _re.MULTILINE
        )
        entity_names = [
            e for e in raw_entities
            if not e.lower().startswith("service")
        ]
        if not entity_names:
            return None

        # ── Entity matching (unchanged) ───────────────────────────────────────
        msg_lower = message.lower()
        best_entity: Optional[str] = None
        for ent in entity_names:
            if ent.lower() in msg_lower or msg_lower in ent.lower():
                best_entity = ent
                break
        if not best_entity:
            msg_words = set(_re.split(r'\W+', msg_lower))
            scored = []
            for ent in entity_names:
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
                overlap = sum(
                    1 for mw in msg_words
                    if any(mw.startswith(stem[:4]) or stem.startswith(mw[:4])
                           for stem in stems if len(stem) >= 4)
                )
                scored.append((overlap, ent))
            scored.sort(reverse=True)
            if scored and scored[0][0] > 0:
                best_entity = scored[0][1]
        if not best_entity and entity_names:
            best_entity = entity_names[0]
        if not best_entity:
            return None

        # ── Build OData query options from NL ─────────────────────────────────
        fields = self._parse_entity_fields(best_entity, schema_hint)
        assocs = self._parse_associations(best_entity, schema_hint)
        odata_filter = self._build_filter(message, fields) if fields else None
        odata_expand = self._build_expand(message, assocs)  if assocs else None

        # Aggregation intent: "by status", "per category", "grouped by type"
        AGG_PAT = _re.compile(
            r'\b(?:by|per|group(?:ed)?\s+by|breakdown\s+by)\s+(\w+)',
            _re.IGNORECASE
        )
        agg_match = AGG_PAT.search(message)
        group_field: Optional[str] = agg_match.group(1) if agg_match else None

        # ── Resolve relative URL ──────────────────────────────────────────────
        if service_url.startswith("/"):
            service_url = f"http://localhost:4004{service_url}"

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
                # Aggregation needs a larger sample; otherwise 5 rows is enough
                top = 200 if group_field else 5
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
                    lines.append(f"First {len(rows_data)} record(s):")
                    for i, row in enumerate(rows_data[:5], 1):
                        preview = {
                            k: v for k, v in row.items()
                            if not isinstance(v, (dict, list)) and v is not None
                        }
                        # Show expanded nav properties as "(N item(s))"
                        for k, v in row.items():
                            if isinstance(v, list) and v is not None:
                                preview[k] = f"({len(v)} item(s))"
                        lines.append(f"  {i}. {_json.dumps(preview, default=str)}")
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

            "4. COUNTS & QUERIES\n"
            "   You cannot directly query the database. For count/list questions, respond with "
            "the correct OData URL from the schema, e.g.: "
            "GET {service_url}/{EntitySetName}/$count\n\n"

            "5. STAY IN APP CONTEXT\n"
            "   Every answer must be grounded in the schema provided. "
            "Do not import entity names or field names from previous conversations or "
            "general SAP knowledge — the current app may have completely different entities."
            + entity_section
        )

        if rag_context:
            base += (
                f"\n\nAdditional retrieved knowledge for {app_label}:\n\n{rag_context}"
            )

        return base

    async def get_response(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        app_id: Optional[str] = None,
        fiori_context: Optional[Dict[str, Any]] = None,
        odata_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.request_count += 1
        start_time = datetime.utcnow()

        rag_context = await self._fetch_rag_context(message, app_id)
        system_message = self._build_system_message(rag_context, app_id, fiori_context)

        # Fetch real live data from the app's OData service when the user
        # is asking a data question (counts, lists, records).
        live_data_block = await self._fetch_real_data(fiori_context, odata_token, message)
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
    ):
        """Stream response token-by-token using word-level chunking."""
        result = await self.get_response(
            message=message,
            history=history,
            app_id=app_id,
            fiori_context=fiori_context,
            odata_token=odata_token,
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
