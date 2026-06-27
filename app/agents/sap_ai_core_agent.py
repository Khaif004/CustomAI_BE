

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

        CAP naming rule: FK fields from parent entities are always named
        'to_<ParentEntity>_<key>' — these are tried first and exclusively
        when present, preventing PK fields (e.g. assignMaterialBlendID) from
        being scored as FK candidates.
        """
        import re as _re
        numbers = _re.findall(r'\b(\d{3,})\b', message)
        if not numbers:
            return None

        to_fk = [
            f for f in fields
            if _re.match(r'(?i)to_', f)
            and _re.search(r'(?i)(orderID|order_id|_ID|ID|Number)$', f)
        ]
        other_fk = [
            f for f in fields
            if not _re.match(r'(?i)to_', f)
            and _re.search(r'(?i)(orderID|order_id|_ID|ID|Number)$', f)
            and f.lower() not in ('id', 'areatype_id', 'crop_id', 'variety_id')
        ]

        fk_fields = to_fk if to_fk else other_fk
        if not fk_fields:
            return None

        msg_lower = message.lower()
        msg_words = set(_re.split(r'\W+', msg_lower))
        best_fk = None
        best_score = -1
        for fk in fk_fields:

            if _re.match(r'(?i)to_', fk):
                m = _re.match(r'(?i)to_(.+)_\w+$', fk)
                score_target = m.group(1) if m else fk
            else:
                score_target = fk
            parts = [p.lower() for p in _re.split(r'(?<=[a-z])(?=[A-Z])|_|(?<=[A-Z])(?=[A-Z][a-z])', score_target) if len(p) > 2]
            score = sum(
                1 for p in parts
                if p in msg_words
                or any(w.startswith(p[:4]) for w in msg_words if len(w) >= 4 and len(p) >= 4)
            )
            if score > best_score:
                best_score = score
                best_fk = fk

        if best_fk:

            _needs_quotes = _re.search(
                r'(?i)(Number|Name|Code|Key|Text|Description|Ref|Reference)$', best_fk
            )
            val = numbers[0]
            return f"{best_fk} eq '{val}'" if _needs_quotes else f"{best_fk} eq {val}"
        return None

    def _build_context_filter(
        self,
        target_fields: list,
        entity_data: dict,
        current_view: str,
    ) -> Optional[str]:
        
        import re as _re
        if not entity_data or not target_fields:
            return None

        _view_entity: Optional[str] = None
        for pat in [
            r'[/#]([A-Z][A-Za-z0-9]+)[/#?]',
            r'[/#]([A-Z][A-Za-z0-9]+)\(',
        ]:
            m = _re.search(pat, current_view or "")
            if m:
                _view_entity = m.group(1)
                break

        _key_candidates: list = []
        for k, v in entity_data.items():
            if v is None:
                continue
            k_lower = k.lower()
            if k_lower.endswith(("id", "key", "number", "no", "code")):
                _key_candidates.append((k, v))
        # Also include bare integers not already listed
        for k, v in entity_data.items():
            if isinstance(v, int) and (k, v) not in _key_candidates:
                _key_candidates.append((k, v))

        def _val_str(v) -> str:
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            return str(v)

        for fk_field in target_fields:
            fk_lower = fk_field.lower()

            # Pass 1: FK field name contains the current view entity name.

            if _view_entity and _view_entity.lower() in fk_lower and "_" in fk_field:
                key_part = fk_field.split("_")[-1]
                for ed_k, ed_v in entity_data.items():
                    if ed_k.lower() == key_part.lower() and ed_v is not None:
                        if isinstance(ed_v, (int, float)):
                            return f"{fk_field} eq {_val_str(ed_v)}"
                        return f"{fk_field} eq '{ed_v}'"

            # Pass 2: FK field name matches an entity_data key exactly.
            for ed_k, ed_v in _key_candidates:
                if fk_lower == ed_k.lower():
                    if isinstance(ed_v, (int, float)):
                        return f"{fk_field} eq {_val_str(ed_v)}"
                    return f"{fk_field} eq '{ed_v}'"

        # Pass 3: cross-entity FK — target entity belongs to a DIFFERENT service

        to_fk_fields = [
            f for f in target_fields
            if _re.match(r'(?i)to_', f) and "_" in f[3:]
        ]
        if to_fk_fields and _key_candidates:
            fk_field = to_fk_fields[0]
            _, val = _key_candidates[0]
            _needs_quotes = _re.search(
                r'(?i)(Number|Name|Code|Key|Text|Description|Ref|Reference)$', fk_field
            )
            val_s = _val_str(val)
            logger.info(
                "[context_filter] pass 3 cross-entity: using fk=%s val=%s quoted=%s",
                fk_field, val_s, bool(_needs_quotes),
            )
            return f"{fk_field} eq '{val_s}'" if _needs_quotes else f"{fk_field} eq {val_s}"

        return None

    def _parse_keys_from_view_url(self, current_view: str) -> dict:
        """Extract key-value pairs from a Fiori hash URL such as
        Returns a dict of numeric/string keys (booleans and 'null' are skipped).
        """
        import re as _re
        result: dict = {}
        m = _re.search(r'[/#]([A-Z][A-Za-z0-9]+)\(([^)]+)\)', current_view or "")
        if not m:
            return result
        for pair in m.group(2).split(','):
            pair = pair.strip()
            if '=' not in pair:
                continue
            k, _, v = pair.partition('=')
            k, v = k.strip(), v.strip()
            if v.lower() in ('true', 'false', 'null', ''):
                continue
            try:
                result[k] = int(v)
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
        return result

    def _build_context_filter_from_view(
        self,
        entity_data: dict,
        current_view: str,
    ) -> Optional[str]:
        """Infer an OData FK filter using CAP naming convention when the field list
        is unavailable (cross-service entity).
        """
        import re as _re
        if not entity_data or not current_view:
            return None

        _view_entity: Optional[str] = None
        for pat in [r'[/#]([A-Z][A-Za-z0-9]+)[/#?]', r'[/#]([A-Z][A-Za-z0-9]+)\(']:
            m = _re.search(pat, current_view)
            if m:
                _view_entity = m.group(1)
                break
        if not _view_entity:
            return None

        def _val_str(v) -> str:
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            return str(v)

        def _try_int(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        # Prefer fields whose name ends with id/key/number/no, then bare ints
        key_candidates: list = []
        for k, v in entity_data.items():
            if v is None or k.startswith("_"):
                continue
            if k.lower().endswith(("id", "key", "number", "no")):
                key_candidates.append((k, v))
        for k, v in entity_data.items():
            if isinstance(v, int) and not k.startswith("_") and (k, v) not in key_candidates:
                key_candidates.append((k, v))

        if not key_candidates:
            return None

        ed_k, ed_v = key_candidates[0]

        target_entity = getattr(self, '_current_fetch_entity', None)
        if target_entity and target_entity.lower() == _view_entity.lower():
            if isinstance(ed_v, (int, float)):
                return f"{ed_k} eq {_val_str(ed_v)}"
            iv = _try_int(ed_v)
            if iv is not None:
                return f"{ed_k} eq {iv}"
            return f"{ed_k} eq '{ed_v}'"

        fk_field = f"to_{_view_entity}_{ed_k}"
        if isinstance(ed_v, (int, float)):
            return f"{fk_field} eq {_val_str(ed_v)}"
        iv = _try_int(ed_v)
        if iv is not None:
            return f"{fk_field} eq {iv}"
        return f"{fk_field} eq '{ed_v}'"

    def _parse_fields_from_rag(self, entity: str, rag_context: str) -> list:
        """Extract field names for a specific entity from RAG context text.

        Handles three source formats:
          • SchemaExtractor  — 'Entity: Name\\nFields: f1 (Type), f2 (Type)'
          • ODataProbe       — '## Name entity schema\\n- f1 (Type)\\n- f2 (Type)'
          • Legacy           — 'Entity: Name ... Available fields: f1, f2'
        Also extracts FK fields from query-guidance lines emitted by SchemaExtractor:
          '$filter=to_FertilizerBlend_orderID%20eq%20<numericValue>'
        """
        import re as _re
        if not rag_context:
            return []

        # ── Locate the entity section ────────────────────────────────────────
        # Match either:
        #   "[title label] Entity: EntityName"  (RAG context wraps each chunk with a label)
        #   "Entity: EntityName"                (plain SchemaExtractor)
        #   "## EntityName ..." or "## Entity: EntityName"  (ODataProbe / markdown)
        section_m = _re.search(
            rf'(?:^|\n)(?:\[[^\]]*\]\s*)?Entity:\s+{_re.escape(entity)}\b'
            rf'(.*?)'
            rf'(?=(?:\n|\A)(?:\[[^\]]*\]\s*)?Entity:\s+[A-Z]|^##\s|\Z)',
            rag_context, _re.DOTALL | _re.IGNORECASE,
        ) or _re.search(
            rf'^##\s+(?:Entity:\s+)?{_re.escape(entity)}(?:\s+entity\s+schema)?\s*$'
            rf'(.*?)'
            rf'(?=^##\s|\Z)',
            rag_context, _re.MULTILINE | _re.DOTALL,
        )
        if not section_m:
            return []

        section = section_m.group(1)
        fields: list = []

        # 1. "Fields: f1 (Type), f2 (Type)" and "Key fields: f1 (Type, key)" (SchemaExtractor)
        for line in section.splitlines():
            m = _re.match(
                r'^(?:Fields|Key\s+fields|Key\s+field|Available\s+fields):\s+(.+)$',
                line.strip(), _re.IGNORECASE,
            )
            if m:
                for item in m.group(1).split(','):
                    fn = _re.match(r'\s*(\w+)\s*[\(\[]', item)
                    if fn and fn.group(1) not in fields:
                        fields.append(fn.group(1))

        # 2. Bullet "- fieldName (Type)" format (ODataProbe)
        if not fields:
            for match in _re.finditer(r'^\s*-\s+(\w+)\s*\(', section, _re.MULTILINE):
                name = match.group(1)
                if name not in fields:
                    fields.append(name)

        # 3. FK fields from query-guidance lines (highest priority for filter building)
        #    SchemaExtractor emits: "$filter=to_FertilizerBlend_orderID%20eq%20<numericValue>"
        #    These exact field names are what _build_fk_filter / _build_context_filter need.
        for m in _re.finditer(r'\$filter=(\w+)%20eq%20', section, _re.IGNORECASE):
            fn = m.group(1)
            if fn not in fields:
                fields.append(fn)

        return fields

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


    async def _llm_plan_fetch(
        self,
        message: str,
        all_entities: list,
        fiori_context: Optional[Dict[str, Any]],
        app_id: Optional[str] = None,
    ) -> list:
        """
        Schema-aware planning agent.

        Builds a compact schema digest from the service tool registry
        (entity names + their actual key/FK/regular fields) and asks the
        LLM to produce a precise OData fetch plan.  Zero hardcoded
        terminology — works for any SAP CAP app because the LLM reasons
        from the real field names registered by that app's cap-plugin.
        """
        import json as _json
        import aiohttp as _aio
        import re as _re

        if not all_entities:
            return []

        current_view = ""
        if fiori_context:
            current_view = fiori_context.get("current_view", "") or ""


        _entity_schema: dict = {}
        if app_id:
            try:
                from app.api.apps import _service_tool_registry
                for svc in _service_tool_registry.get(app_id, []):
                    for ent, flds in (svc.get("entity_fields") or {}).items():
                        fk_fields = [f for f in flds if f.lower().startswith("to_")]
                        key_fields = [
                            f for f in flds
                            if not f.lower().startswith("to_")
                            and _re.search(r'(?i)(^id$|ID$|Key$|Number$|Code$)', f)
                            and f not in ("IsActiveEntity", "HasActiveEntity", "HasDraftEntity")
                        ]
                        sample = [
                            f for f in flds
                            if f not in fk_fields and f not in key_fields
                            and f not in ("createdAt", "createdBy", "modifiedAt", "modifiedBy",
                                          "IsActiveEntity", "HasActiveEntity", "HasDraftEntity")
                        ][:4]
                        _entity_schema[ent] = {"keys": key_fields, "fks": fk_fields, "sample": sample}
            except Exception:
                pass


        _schema_ent_names = [ent_name for ent_name, _ in all_entities[:60]]
        _ent_names_lower = [e.lower() for e in _schema_ent_names]
        _deduped_entities = []
        for ent_name in _schema_ent_names:
            _bare = ent_name.lower()
            _bare_s = _bare[:-1] if _bare.endswith('s') else _bare
            _has_compound = any(
                e != _bare and (e.endswith(_bare) or e.endswith(_bare_s) or e.endswith(_bare_s + 's'))
                for e in _ent_names_lower
            )
            if _has_compound:
                continue
            _deduped_entities.append(ent_name)

        schema_lines = []
        for ent_name in _deduped_entities:
            s = _entity_schema.get(ent_name, {})
            # If this entity lacks schema, check if a compound variant has it
            if not s:
                _bare = ent_name.lower()
                for _reg_ent, _reg_s in _entity_schema.items():
                    if _reg_ent.lower() != _bare and (
                        _reg_ent.lower().endswith(_bare)
                        or _reg_ent.lower().endswith(_bare[:-1] if _bare.endswith('s') else _bare)
                    ):
                        s = _reg_s
                        break
            parts = []
            if s.get("keys"):   parts.append(f"keys=[{', '.join(s['keys'][:3])}]")
            if s.get("fks"):    parts.append(f"fks=[{', '.join(s['fks'][:4])}]")
            if s.get("sample"): parts.append(f"fields=[{', '.join(s['sample'][:3])}]")
            suffix = f"  ({'; '.join(parts)})" if parts else ""
            schema_lines.append(f"  - {ent_name}{suffix}")

        schema_block = "\n".join(schema_lines)

        # ── System prompt — fully schema-driven, no hardcoded app terminology ─
        planner_system = (
            "You are a schema-aware OData fetch planner for SAP CAP applications.\n"
            "You receive the app's entity schema (entity names, key fields, FK fields)\n"
            "and the user question.  Produce a precise OData fetch plan.\n\n"

            "RULES:\n"
            "1. ENTITY SELECTION — select the entity for what the user WANTS TO SEE,\n"
            "   not the entity named as context/parent.\n"
            "   Pattern: '<child-things> for/of <parent>' → pick the CHILD entity.\n"
            "   Pattern: 'show me <entity>'               → pick that entity directly.\n"
            "   When both a parent entity (e.g. SalesOrder) and a child entity\n"
            "   (e.g. SalesOrderItem) exist in the schema, and the user asks for\n"
            "   'items', 'lines', 'details', or similar, always pick the CHILD.\n\n"
            "2. FILTER — derive the correct field from the schema:\n"
            "   a) Child entity (has FK fields in fks=[...]): use the FK field name\n"
            "      EXACTLY as it appears in fks=[...] — never invent your own FK name.\n"
            "      Example: if fks=[to_SalesOrder_salesOrderNumber] → filter must be\n"
            "      'to_SalesOrder_salesOrderNumber eq <value>'. Do NOT substitute field\n"
            "      names from the URL hash (e.g. orderID) into the FK pattern.\n"
            "   b) Top-level entity (user references it directly by key): use its own\n"
            "      key field from keys=[...]. NEVER write a 'to_<Self>_...' filter.\n\n"
            "3. VALUE QUOTING (from the key/FK field name in the schema):\n"
            "   - Field ends with 'ID' or 'Id' or value is a plain integer → no quotes\n"
            "   - Field ends with 'Number', 'Code', 'Name', 'Key', 'Text' → single quotes\n\n"
            "4. CONTEXT KEY — extract the key value from current_view URL when user\n"
            "   says 'this', 'current', 'for this', etc.\n\n"
            "5. MULTI-ENTITY (CRITICAL) — when the user asks for MULTIPLE things in\n"
            "   one question (uses 'and', 'also', commas, or lists multiple entity names),\n"
            "   you MUST return one fetch entry per requested concept — up to 5 total.\n"
            "   Examples:\n"
            "     'farms and fields and areas' → 3 entries (SelectedFarms, SelectedFields, SelectedAreas)\n"
            "     'show me X, Y and Z' → 3 entries\n"
            "   NEVER collapse multiple requested entities into one single entry.\n\n"
            "6. Return {\"fetches\":[]} for greetings / general / how-to questions.\n\n"
            "7. When uncertain about the filter field, set filter to null.\n\n"
            "RESPOND WITH ONLY valid JSON (no markdown):\n"
            "{\"fetches\": [{\"entity\": \"EntityName\", \"filter\": \"OData expr or null\"}]}"
        )

        planner_user = (
            f"App entity schema:\n{schema_block}\n\n"
            f"Current view URL: {current_view or '(none)'}\n\n"
            f"User question: {message}"
        )

        try:
            token = await self.auth_client.get_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "AI-Resource-Group": "default",
            }
            payload = {
                "orchestration_config": {
                    "module_configurations": {
                        "llm_module_config": {
                            "model_name": self.model_id,
                            "model_params": {
                                "max_tokens": 512,
                                "temperature": 0.0,
                            },
                        },
                        "templating_module_config": {
                            "template": [
                                {"role": "system", "content": planner_system},
                                {"role": "user",   "content": "{{?q}}"},
                            ]
                        },
                    }
                },
                "input_params": {"q": planner_user},
            }

            async with _aio.ClientSession() as session:
                async with session.post(
                    self.inference_url,
                    json=payload,
                    headers=headers,
                    timeout=_aio.ClientTimeout(total=20),
                ) as resp:
                    if resp.status != 200:
                        logger.warning("[planner] LLM call failed: %s", resp.status)
                        return []
                    result = await resp.json()

            # Extract content from orchestration response
            content = ""
            mr = result.get("module_results", {}).get("llm", {})
            if mr.get("choices"):
                content = mr["choices"][0].get("message", {}).get("content", "")
            elif "orchestration_result" in result:
                choices = result["orchestration_result"].get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            plan = _json.loads(content)
            fetches = plan.get("fetches", [])
            logger.info("[planner] plan: %s", fetches)
            return fetches

        except Exception as e:
            logger.warning("[planner] failed (%s) — falling back to regex", e)
            return []


    async def _fetch_real_data(
        self,
        fiori_context: Optional[Dict[str, Any]],
        odata_token: Optional[str],
        message: str,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        rag_context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        planned_entity: Optional[str] = None,
        planned_filter: Optional[str] = None,
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
            r"|list|show|display|give|also"
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
        # "what are the MATERIALS for..."   → primary = "materials"
        # "I need all the saleorderitems..." → primary = "saleorderitems"
        # "give me the blends..."            → primary = "blends"
        # Used to (a) constrain Pass 1 exact-match and (b) boost scoring in Pass 2.
        _pn_match = _re.search(
            r'\b(?:what\s+(?:are\s+)?(?:the\s+)?'
            r'|show\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?'
            r'|list\s+(?:all\s+)?(?:the\s+)?'
            r'|get\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?'
            r'|fetch\s+(?:all\s+)?(?:the\s+)?'
            r'|find\s+(?:all\s+)?(?:the\s+)?'
            r'|need\s+(?:all\s+)?(?:the\s+)?'           # "I need all the X"
            r'|give\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?'  # "give the X" / "give me the X"
            r'|tell\s+me\s+(?:about\s+)?(?:the\s+)?'   # "tell me about the X"
            r'|looking\s+for\s+(?:all\s+)?(?:the\s+)?'  # "looking for the X"
            r'|also\s+(?:give\s+)?(?:the\s+)?'          # "also give the X" / "also the X"
            r'|can\s+you\s+(?:also\s+)?(?:give\s+)?(?:the\s+)?'  # "can you also give the X"
            r')\s*(\w+)',
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

        if planned_entity:
            _pe_lower = planned_entity.lower()
            for ent in all_candidate_entities:
                if ent.lower() == _pe_lower:
                    best_entity = ent
                    best_service_url = _entity_to_service.get(ent, service_url)
                    break
            if not best_entity:
                best_entity = planned_entity
                best_service_url = _entity_to_service.get(planned_entity, service_url)
            logger.info("[fetch_real_data] planner-resolved entity=%s filter=%s", best_entity, planned_filter)

        # Pass 1: exact containment — scoped to PRIMARY NOUN when identifiable.
        # Rationale: "what are the saleorderitems for salesorder 2466?" contains
        # "salesorder" as a FILTER word, not the main subject.  If we allowed full-message
        # scanning, SalesOrder would win over SalesOrderItems.  By restricting Pass 1 to
        # only match when the entity name == the primary noun, we force compound-word queries
        # into Pass 2 where trigram + stem-count scoring correctly discriminates sub-entities.
        for ent in all_candidate_entities:
            ent_lower = ent.lower()
            if _primary_noun:
                # Accept exact match OR plural variant (e.g. "processorders" → ProcessOrder).
                _pn_singular = _primary_noun[:-1] if _primary_noun.endswith('s') else _primary_noun
                if ent_lower == _primary_noun or ent_lower == _pn_singular:
                    best_entity = ent
                    best_service_url = _entity_to_service.get(ent, service_url)
                    break
            else:
                # No primary noun detected → original full-message containment.
                if ent_lower in msg_lower:
                    best_entity = ent
                    best_service_url = _entity_to_service.get(ent, service_url)
                    break


        if best_entity:
            _be_lower = best_entity.lower()
            for ent in all_candidate_entities:
                if ent == best_entity:
                    continue
                ent_lower = ent.lower()
                if not ent_lower.endswith(_be_lower) and not ent_lower.endswith(_be_lower[:-1] if _be_lower.endswith('s') else _be_lower + 's'):
                    continue
                _extra_parts = _re.findall(r'[A-Z][a-z0-9]+', ent[:len(ent) - len(best_entity)])
                if _extra_parts and all(p.lower() in msg_lower for p in _extra_parts):
                    best_entity = ent
                    best_service_url = _entity_to_service.get(ent, service_url)
                    logger.info(
                        "[fetch_real_data] Pass 1b: refined '%s' → compound entity '%s'",
                        _be_lower, ent,
                    )
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

                # Primary-noun boost: prefer entities whose name contains MORE stems
                # from the primary noun.  Counts how many entity word-stems appear as
                # substrings inside the primary noun (handles compound typos).
                # Example: primary="saleorderitems"
                #   SalesOrderItems → stems sale✓ order✓ item✓ → boost = 3×10 = 30
                #   SalesOrder      → stems sale✓ order✓       → boost = 2×10 = 20
                # → SalesOrderItems correctly wins.
                primary_boost = 0.0
                if _primary_noun and len(_primary_noun) >= 4:
                    pn_lower = _primary_noun.lower()
                    stem_hits = sum(
                        1 for s in stems
                        if len(s) >= 4 and (
                            s in pn_lower                      # stem is substring of noun
                            or pn_lower.startswith(s[:4])      # noun starts with stem prefix
                            or s.startswith(pn_lower[:4])      # stem starts with noun prefix
                        )
                    )
                    primary_boost = stem_hits * 10.0

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


        self._current_fetch_entity = best_entity

        # ── Build OData query options from NL ─────────────────────────────────
        # Priority: registry field map → schema_hint → RAG context
        # For cross-service entities the widget's schema_hint won't cover them;
        # for standalone chat schema_hint is empty — always fall back to RAG.
        if not _cross_service:
            fields = self._parse_entity_fields(best_entity, schema_hint)
            assocs = self._parse_associations(best_entity, schema_hint)
            # schema_hint may not include this entity (different entity in current view,
            # or standalone chat with no widget) — fall back to RAG in that case.
            if not fields and rag_context:
                fields = self._parse_fields_from_rag(best_entity, rag_context)
        else:
            fields = self._parse_fields_from_rag(best_entity, rag_context or "")
            assocs = []

        # Last resort: look up fields from the service tool registry (sent by cap-plugin)
        if not fields and app_id:
            try:
                from app.api.apps import _service_tool_registry
                for svc in _service_tool_registry.get(app_id, []):
                    reg_fields = svc.get("entity_fields", {}).get(best_entity, [])
                    if reg_fields:
                        fields = reg_fields
                        break
            except Exception:
                pass

        logger.info(
            "[fetch_real_data] fields resolved: count=%d cross_service=%s sample=%s",
            len(fields or []), _cross_service, (fields or [])[:4],
        )

        # ── Planner-provided filter takes full priority ───────────────────────
        if planned_filter:
            odata_filter = planned_filter
            odata_expand = self._build_expand(message, assocs) if assocs else None
        else:
            odata_filter = self._build_filter(message, fields) if fields else None
            if not odata_filter:
                # FK filter: try to match a bare number in the message to a FK field
                odata_filter = self._build_fk_filter(message, fields) if fields else None
            if not odata_filter and fiori_context:
                # Context-aware FK filter: "linked to this blend" has no number in the
                # message — the ID (e.g. orderID=2466) lives in entity_data from the
                # current Fiori view. Use it to build to_FertilizerBlend_orderID eq 2466.
                _CONTEXT_REF = _re.compile(
                    r"\b(this|current|linked\s+to|associated\s+with|for\s+this|"
                    r"in\s+this|on\s+this|of\s+this|belonging\s+to)\b",
                    _re.IGNORECASE,
                )
                if _CONTEXT_REF.search(message):
                    _entity_data = fiori_context.get("entity_data") or {}
                    _current_view = fiori_context.get("current_view", "")
                    # When entity_data is empty but the hash URL carries key params
                    # (e.g. #/FertilizerBlend(orderID=2414,IsActiveEntity=true)),
                    # parse the numeric keys directly from the URL so the filter
                    # can still be built even when the frontend sent no entity_data.
                    if not _entity_data and _current_view:
                        _entity_data = self._parse_keys_from_view_url(_current_view)
                        if _entity_data:
                            logger.info(
                                "[fetch_real_data] entity_data was empty — parsed from view URL: %s",
                                _entity_data,
                            )
                    logger.info(
                        "[fetch_real_data] context filter check: entity_data_keys=%s "
                        "current_view=%r fields_count=%d",
                        list(_entity_data.keys())[:6], _current_view, len(fields or []),
                    )
                    _ctx_filter: Optional[str] = None
                    if _entity_data and fields:
                        _ctx_filter = self._build_context_filter(
                            fields, _entity_data, _current_view
                        )
                    # Fallback: fields unavailable (cross-service) or _build_context_filter
                    # returned None — infer FK from CAP to_<Entity>_<key> naming convention.
                    if not _ctx_filter and _entity_data and _current_view:
                        _ctx_filter = self._build_context_filter_from_view(
                            _entity_data, _current_view
                        )
                        if _ctx_filter:
                            logger.info(
                                "[fetch_real_data] context filter (view-inferred): %s",
                                _ctx_filter,
                            )
                    if _ctx_filter:
                        odata_filter = _ctx_filter
                        logger.info(
                            "[fetch_real_data] context-aware filter from entity_data: %s",
                            odata_filter,
                        )
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

        # ── Row limits driven by settings (no hardcoding) ────────────────────
        try:
            from app.config.settings import get_settings as _gs
            _cfg = _gs()
            _DISPLAY_ROWS: int = _cfg.odata_display_rows    # shown in chat (default 10)
            _FETCH_ROWS: int   = _cfg.odata_fetch_rows      # fetched from OData (default 50)
            _AGG_ROWS: int     = _cfg.odata_aggregate_rows  # for group-by (default 500)
        except Exception:
            _DISPLAY_ROWS, _FETCH_ROWS, _AGG_ROWS = 10, 50, 500

        # Aggregation intent: "by status", "per category", "grouped by type"
        AGG_PAT = _re.compile(
            r'\b(?:by|per|group(?:ed)?\s+by|breakdown\s+by)\s+(\w+)',
            _re.IGNORECASE
        )
        agg_match = AGG_PAT.search(message)
        group_field: Optional[str] = agg_match.group(1) if agg_match else None

        # ── Resolve relative service URL ──────────────────────────────────────
        # Priority order for the CAP server base URL:
        #   1. app_base_url stored in registry by cap-plugin at startup (always preferred)
        #   2. cap_app_base_url from settings  (set in .env for cloud deployments)
        #   3. Empty string → service_url must already be absolute (logged as warning)
        if service_url.startswith("/"):
            if _registry_base_url:
                base = _registry_base_url.rstrip("/")
            else:
                try:
                    from app.config.settings import get_settings as _gs2
                    _cap_base = _gs2().cap_app_base_url.rstrip("/")
                except Exception:
                    _cap_base = ""
                if _cap_base:
                    base = _cap_base
                else:
                    logger.warning(
                        "[fetch_real_data] relative service_url '%s' but no base URL in registry "
                        "or settings.cap_app_base_url — set CAP_APP_BASE_URL in .env",
                        service_url,
                    )
                    base = ""
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

            # aiohttp default urlencode uses quote_plus (spaces → '+'), but CAP's
            # OData parser only accepts %20.  Build query strings manually.
            def _odata_qs(params: dict) -> str:
                from urllib.parse import urlencode, quote as _q
                return urlencode({k: str(v) for k, v in params.items()}, quote_via=_q)

            # ── Resolve the entity SET name (may differ from type name) ───────
            # ODataProbe captures type names (SalesOrderItem); the OData endpoint
            # uses set names (SalesOrderItems).  Prefer the registry-registered
            # name when available — it is always the correct entity set name.
            def _resolve_set_name(type_name: str) -> str:
                if type_name in _entity_to_service:      # already a set name
                    return type_name
                # Try adding/removing 's'
                plural = type_name + 's'
                if plural in _entity_to_service:
                    logger.info("[fetch_real_data] Resolved '%s' → set name '%s'", type_name, plural)
                    return plural
                singular = type_name[:-1] if type_name.endswith('s') else type_name
                if singular in _entity_to_service:
                    logger.info("[fetch_real_data] Resolved '%s' → set name '%s'", type_name, singular)
                    return singular
                return type_name  # fall back to type name and let OData 404 trigger retry

            entity_set_name = _resolve_set_name(best_entity)
            base_url = f"{service_url.rstrip('/')}/{entity_set_name}"
            count_val: Optional[int] = None
            rows_data = None
            # Candidate URL names to try in order on 404: resolved set name → original → plural → singular
            _url_candidates: list = [entity_set_name]
            if best_entity != entity_set_name:
                _url_candidates.append(best_entity)
            if not entity_set_name.endswith('s'):
                _url_candidates.append(entity_set_name + 's')
            elif entity_set_name.endswith('s'):
                _url_candidates.append(entity_set_name[:-1])

            async with _aio.ClientSession() as session:
                # Aggregation needs a much larger sample; otherwise use _FETCH_ROWS
                top = _AGG_ROWS if group_field else _FETCH_ROWS
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
                    _count_url = f"{base_url}/$count?{_odata_qs(count_params)}" if count_params else f"{base_url}/$count"
                    async with session.get(
                        _count_url,
                        headers=headers, timeout=_aio.ClientTimeout(total=8),
                    ) as resp:
                        if resp.status == 200:
                            try:
                                count_val = int((await resp.text()).strip())
                            except ValueError:
                                pass
                        elif resp.status == 404:
                            # Retry with alternate entity set name spellings (404 = wrong name)
                            for _alt in _url_candidates[1:]:
                                _alt_url = f"{service_url.rstrip('/')}/{_alt}"
                                _alt_count_url = f"{_alt_url}/$count?{_odata_qs(count_params)}" if count_params else f"{_alt_url}/$count"
                                async with session.get(
                                    _alt_count_url,
                                    headers=headers, timeout=_aio.ClientTimeout(total=8),
                                ) as r2:
                                    if r2.status == 200:
                                        base_url = _alt_url
                                        entity_set_name = _alt
                                        logger.info("[fetch_real_data] 404 retry succeeded with '%s'", _alt)
                                        try:
                                            count_val = int((await r2.text()).strip())
                                        except ValueError:
                                            pass
                                        break

                # Fetch rows (with optional filter + expand)
                _list_url = f"{base_url}?{_odata_qs(list_params)}"
                async with session.get(
                    _list_url,
                    headers=headers, timeout=_aio.ClientTimeout(total=8),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        rows_data = data.get("value", [])
                        if count_val is None and not group_field:
                            total = data.get("@odata.count")
                            if total is not None:
                                count_val = int(total)
                    elif resp.status == 404 and base_url.endswith(entity_set_name):
                        # Last-resort retry with alternate entity name (404 = wrong set name)
                        safe_params: dict = {"$top": top}
                        if odata_filter:
                            safe_params["$filter"] = odata_filter
                        for _alt in _url_candidates[1:]:
                            _alt_url = f"{service_url.rstrip('/')}/{_alt}"
                            async with session.get(
                                f"{_alt_url}?{_odata_qs(safe_params)}",
                                headers=headers, timeout=_aio.ClientTimeout(total=8),
                            ) as r2:
                                if r2.status == 200:
                                    data = await r2.json()
                                    rows_data = data.get("value", [])
                                    logger.info("[fetch_real_data] rows 404 retry succeeded with '%s'", _alt)
                                    break

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
                total = count_val or len(rows_data or [])
                if count_val is not None:
                    lines.append(f"Total record count: {count_val}")

                if rows_data:
                    # ── Column ordering: name/ID fields first ─────────────────
                    uuid_re = _re.compile(
                        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                        _re.IGNORECASE
                    )
                    # Collect all scalar, non-UUID columns across sample rows
                    _all_keys: list = []
                    for row in rows_data[:_DISPLAY_ROWS]:
                        for k in row:
                            if k not in _all_keys:
                                _all_keys.append(k)

                    scalar_keys = [
                        k for k in _all_keys
                        if not isinstance(rows_data[0].get(k), (dict, list))
                    ]
                    meaningful_keys = [
                        k for k in scalar_keys
                        if not (
                            isinstance(rows_data[0].get(k), str)
                            and uuid_re.match(str(rows_data[0].get(k, "")))
                            and k.lower() not in ("id", "key")
                        )
                    ] or scalar_keys

                    # Sort: name/title/number/id columns first for readability
                    def _col_priority(col: str) -> int:
                        cl = col.lower()
                        if cl in ("id", "key"):                         return 0
                        if cl.endswith("id") or cl.endswith("_id"):     return 1
                        if "number" in cl:                              return 2
                        if "name" in cl or "title" in cl:               return 3
                        if "description" in cl or "desc" in cl:         return 4
                        if "status" in cl or "state" in cl:             return 5
                        if "type" in cl or "category" in cl:            return 6
                        if "date" in cl or "at" in cl:                  return 8
                        if "by" in cl:                                  return 9
                        return 7

                    meaningful_keys.sort(key=_col_priority)

                    # ── Display limit + export offer ──────────────────────────
                    # When a server-side $filter was applied, show ALL fetched rows
                    # (they are already scoped to exactly what the user asked for).
                    # Only cap unfiltered results to avoid overwhelming the LLM with
                    # an entire unrelated entity dump.
                    display = rows_data if odata_filter else rows_data[:_DISPLAY_ROWS]
                    has_more = count_val is not None and count_val > len(display)

                    lines.append(
                        f"Showing {len(display)} of {total} record(s):"
                        if has_more else
                        f"Records ({len(display)}):"
                    )

                    if meaningful_keys:
                        lines.append("| " + " | ".join(meaningful_keys) + " |")
                        lines.append("|" + "|".join("---" for _ in meaningful_keys) + "|")
                        for row in display:
                            cells = [str(row.get(k, "")) for k in meaningful_keys]
                            lines.append("| " + " | ".join(cells) + " |")
                    else:
                        for i, row in enumerate(display, 1):
                            lines.append(f"  {i}. {_json.dumps(row, default=str)}")

                    # ── Hint when there are more records ──────────────────
                    if has_more:
                        lines.append("")
                        lines.append(
                            f"*(Showing {len(display)} of {total} records total. "
                            f"Let me know if you'd like a full report — "
                            f"I can generate an Excel, PDF, Word, or CSV document with all the data.)*"
                        )

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

            "3b. DATA UNAVAILABILITY\n"
            "   If no '[Real data from OData]' block is present in the message:\n"
            "   a) Do NOT invent example rows, fake field values, or placeholder tables.\n"
            "   b) Do NOT give raw OData URLs to business users.\n"
            "   c) Do NOT just say 'try again' or 'I wasn't able to retrieve' without more help.\n"
            "   d) Explain WHAT entity holds the data the user is asking for (name it from the schema),\n"
            "      and ask one targeted clarifying question if needed (e.g. which order ID?).\n"
            "      Example: 'The item-level details live in the SalesOrderItem entity. "
            "I couldn't retrieve those records \u2014 could you confirm order number 2466 is correct?'\n\n"

            "4. COUNTS & LIVE DATA\n"
            "   The user's message may be prefixed with a '[Real data from OData \u2014 EntityName]' "
            "block. When that block is present:\n"
            "   a) Use the exact numbers from it \u2014 do NOT say you cannot query the database.\n"
            "   b) State the total count naturally (e.g. 'There are 4 fertilizer blends\u2026').\n"
            "   c) Present ALL sample records as a **Markdown table** \u2014 never show just 1 row when\n"
            "      multiple rows are in the block. Use the most meaningful scalar fields as columns.\n"
            "   d) After the table, add 1-2 sentences of insight (status breakdown, date range, etc.).\n"
            "   e) If the live data block is absent, follow rule 3b.\n\n"

            "5. STAY IN APP CONTEXT\n"
            "   Every answer must be grounded in the schema provided. "
            "Do not import entity names or field names from previous conversations or "
            "general SAP knowledge \u2014 the current app may have completely different entities.\n\n"

            "7. PARENT \u2192 CHILD NAVIGATION\n"
            "   When a user asks for child/related records (items, materials, components, lines):\n"
            "   a) Identify the CHILD entity from the schema \u2014 it will have a FK field referencing the parent.\n"
            "   b) A live-data block for the child entity filtered by parent ID will be provided.\n"
            "      Present it as a COMPLETE table \u2014 do not summarise to just the first row.\n"
            "   c) If you receive parent-entity data but the user asked for child data, say:\n"
            "      'The [child entity] data for this [parent] is separate \u2014 try asking: "
            "      \"show me the [child] for [parent name/ID] [value]\"'\n\n"

            "8. COMPLETENESS\n"
            "   When the user says 'all', 'full', 'complete', or 'not just parent':\n"
            "   a) Show the full table from the live-data block (up to 20 rows).\n"
            "   b) If total > 20, state 'showing {n} of {total}' below the table.\n"
            "   c) Never silently truncate or show fewer rows than are in the live-data block."
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
            "  \u2022 Lead with the direct answer \u2014 never start with 'Sure!' / 'Great!' / 'Of course!'.\n"
            "  \u2022 Use **Markdown tables** for any set of records (>1 row).\n"
            "  \u2022 Use numbered lists for ordered steps; bullet points for unordered options.\n"
            "  \u2022 Use **bold** for entity/field names and key values.\n"
            "  \u2022 Keep responses concise. Expand only if the user asked for detail.\n"
            "  \u2022 When you show a count, follow with a one-line summary (e.g. breakdown by status).\n"
            "  \u2022 Never repeat the user's question back to them.\n"
            "  \u2022 Do not add disclaimers like 'I don't have access to real-time data' when a "
            "[Real data from OData] block is present \u2014 you clearly DO have real data.\n\n"
            "DOCUMENT GENERATION RULE:\n"
            "  When the user asks to generate a report, document, Excel, PDF, Word, or CSV:\n"
            "  \u2022 Confirm what data you will include and generate it immediately.\n"
            "  \u2022 Do NOT explain the file format or give instructions on how to open it.\n"
            "  \u2022 Do NOT say the file is 'ready to download' \u2014 just confirm it is being generated."
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
        backend_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.request_count += 1
        start_time = datetime.utcnow()

        rag_context = await self._fetch_rag_context(message, app_id)
        system_message = self._build_system_message(rag_context, app_id, fiori_context, user_id)

        _match_msg = raw_message if raw_message else message

        # Build entity list for the planner from the service tool registry
        _plan_entities: list = []
        if app_id:
            try:
                from app.api.apps import _service_tool_registry
                for _svc in _service_tool_registry.get(app_id, []):
                    for _ent in _svc.get("entities", []):
                        _plan_entities.append((_ent, _svc.get("service_url", "")))
            except Exception:
                pass

        live_data_block: Optional[str] = None

        if _plan_entities:
            _fetch_plan = await self._llm_plan_fetch(_match_msg, _plan_entities, fiori_context, app_id=app_id)
        else:
            _fetch_plan = []

        if _fetch_plan:

            import re as _plan_re

            _reg_flds_lower: dict = {}
            _reg_flds_orig:  dict = {}
            if app_id:
                try:
                    from app.api.apps import _service_tool_registry
                    for _sv in _service_tool_registry.get(app_id, []):
                        for _en, _flist in (_sv.get("entity_fields") or {}).items():
                            _reg_flds_orig.setdefault(_en, []).extend(_flist)
                            lset = _reg_flds_lower.setdefault(_en, set())
                            for _f in _flist:
                                lset.add(_f.lower())
                except Exception:
                    pass

            def _apply_quoting(fk_field: str, val_clean: str) -> str:
                """Return 'fk eq value' with single-quotes iff the field name ends
                in a string-typed suffix per CAP convention."""
                _needs_q = _plan_re.search(
                    r'(?i)(Number|Name|Code|Key|Text|Description|Ref|Reference)$',
                    fk_field,
                )
                return (
                    f"{fk_field} eq '{val_clean}'"
                    if _needs_q
                    else f"{fk_field} eq {val_clean}"
                )

            for _item in _fetch_plan:
                _pf_raw = (_item.get("filter") or "").strip()
                if not _pf_raw:
                    continue
                _pe_raw = (_item.get("entity") or "").strip()
                _ent_flds_lower = _reg_flds_lower.get(_pe_raw, set())
                _ent_flds_orig  = _reg_flds_orig.get(_pe_raw, [])

                # ── Pass 1b: resolve compound entity (mirrors _fetch_real_data logic) ──
                # If LLM returned 'Farms' but registry only has 'SelectedFarms',
                # resolve it now so filter validation/repair uses the right FK fields.
                if not _ent_flds_lower:
                    _pe_lower = _pe_raw.lower()
                    _pe_singular = _pe_lower[:-1] if _pe_lower.endswith('s') else _pe_lower
                    for _reg_ent in list(_reg_flds_orig.keys()):
                        _re_lower = _reg_ent.lower()
                        if (
                            _re_lower != _pe_lower
                            and (
                                _re_lower.endswith(_pe_lower)
                                or _re_lower.endswith(_pe_singular)
                                or _re_lower.endswith(_pe_lower + 's')
                            )
                        ):
                            _ent_flds_lower = _reg_flds_lower.get(_reg_ent, set())
                            _ent_flds_orig  = _reg_flds_orig.get(_reg_ent, [])
                            logger.info(
                                "[planner] Pass 1b resolved '%s' → '%s' for filter validation",
                                _pe_raw, _reg_ent,
                            )
                            _item["entity"] = _reg_ent
                            _pe_raw = _reg_ent
                            break

                _ff_m = _plan_re.match(
                    r'(\w+)\s+(?:eq|ne|lt|gt|le|ge)\s+(.+)',
                    _pf_raw, _plan_re.IGNORECASE,
                )
                if not _ff_m:
                    continue
                _ff_name  = _ff_m.group(1)
                _ff_val   = _ff_m.group(2).strip()
                _ff_clean = _ff_val.strip("'\"")

                if _pe_raw and _plan_re.match(
                    rf'(?i)to_{_plan_re.escape(_pe_raw)}_', _ff_name
                ):
                    logger.warning(
                        "[planner] self-referencing FK stripped: entity=%s filter=%r",
                        _pe_raw, _pf_raw,
                    )
                    _item["filter"] = None
                    continue

                if _ent_flds_lower and _ff_name.lower() in _ent_flds_lower:
                    continue

                if not _ent_flds_lower:
                    logger.warning(
                        "[planner] no schema for entity '%s' — keeping filter %r as-is "
                        "(schema will be available after cap-copilot-sdk pushes metadata)",
                        _pe_raw, _pf_raw,
                    )
                    # Do NOT strip the filter — let OData validate it.
                    continue

                _repaired = False

                _fk_parent_m = _plan_re.match(
                    r'to_(\w+)_\w+', _ff_name, _plan_re.IGNORECASE
                )
                if _fk_parent_m:
                    _bad_parent = _fk_parent_m.group(1).lower()
                    _candidates = [
                        f for f in _ent_flds_orig
                        if f.lower().startswith("to_") and _bad_parent in f.lower()
                    ]
                    if _candidates:
                        _new_filter = _apply_quoting(_candidates[0], _ff_clean)
                        logger.info(
                            "[planner] repaired FK suffix: %r → %r (entity=%s)",
                            _pf_raw, _new_filter, _pe_raw,
                        )
                        _item["filter"] = _new_filter
                        _repaired = True

                if not _repaired:
                    _bare = _ff_name.lower()
                    _candidates = [
                        f for f in _ent_flds_orig
                        if f.lower().startswith("to_") and f.lower().endswith(f"_{_bare}")
                    ]
                    if _candidates:
                        _new_filter = _apply_quoting(_candidates[0], _ff_clean)
                        logger.info(
                            "[planner] repaired bare key → FK: %r → %r (entity=%s)",
                            _pf_raw, _new_filter, _pe_raw,
                        )
                        _item["filter"] = _new_filter
                        _repaired = True

                if not _repaired:
                    logger.warning(
                        "[planner] field '%s' not in schema for '%s' — "
                        "stripping filter %r", _ff_name, _pe_raw, _pf_raw,
                    )
                    _item["filter"] = None

            # Execute each planned fetch in PARALLEL and collect non-empty blocks
            import asyncio as _asyncio
            _fetch_tasks = [
                self._fetch_real_data(
                    fiori_context, odata_token, _match_msg,
                    user_id, app_id,
                    rag_context=rag_context,
                    history=history,
                    planned_entity=_item.get("entity"),
                    planned_filter=_item.get("filter") or None,
                )
                for _item in _fetch_plan if _item.get("entity")
            ]
            _results = await _asyncio.gather(*_fetch_tasks, return_exceptions=True)
            _blocks = [r for r in _results if isinstance(r, str) and r]
            if _blocks:
                live_data_block = "\n\n".join(_blocks)


        _planner_ran = bool(_fetch_plan)
        if not live_data_block and not _planner_ran:
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
        backend_url: Optional[str] = None,
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
            backend_url=backend_url,
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
