"""NavigationAgent — rule-based in-app navigation for CAP Fiori apps.

Handles messages like:
  "open blend 1856"
  "navigate to blend 2144"
  "go to FertilizerBlend 1856"
  "show me the fertilizer blend list"

No LLM call is made. Entity names are resolved from the service tool registry
so the agent works for any registered CAP app without configuration.

The agent emits a BTP_NAVIGATE tool_result SSE event. The SDK widget picks this
up via postMessage and dispatches a CustomEvent that the Fiori controller handles.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Intent detection ─────────────────────────────────────────────────────────

_NAV_VERBS = (
    r"(?:open|navigate\s+to|go\s+to|show\s+me|display|take\s+me\s+to|view|load|"
    r"bring\s+up|jump\s+to|switch\s+to|move\s+to|visit|access)"
)
_NAV_RE = re.compile(rf"(?i)\b{_NAV_VERBS}\b")

# Primary key: 3+ digit integer
_KEY_RE = re.compile(r"\b(\d{3,})\b")


def _camel_words(name: str) -> list[str]:
    """Split "FertilizerBlend" → ["fertilizer", "blend", "fertilizerblend"]."""
    parts = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name).split()
    parts_lower = [p.lower() for p in parts]
    return parts_lower + [name.lower()]


class NavigationAgent:
    """Lightweight rule-based agent for in-app navigation intents."""

    # ── Intent detection ──────────────────────────────────────────────────────

    @staticmethod
    def is_navigation_intent(message: str) -> bool:
        """Quick check — does this message look like a navigation request?"""
        return bool(_NAV_RE.search(message))

    # ── Entity resolution ─────────────────────────────────────────────────────

    def _entity_names(self, app_id: str) -> list[str]:
        """Return all entity names registered for *app_id* (best-effort)."""
        try:
            from app.api.apps import _service_tool_registry
            seen: set[str] = set()
            for svc in _service_tool_registry.get(app_id, []):
                for ent in (svc.get("entity_fields") or {}):
                    seen.add(ent)
            return list(seen)
        except Exception:
            return []

    def _match_entity(self, message: str, entities: list[str]) -> str | None:
        """Find the best-matching entity name from a free-text message.

        Scoring (higher wins):
          +20  — entity name (or last camelCase segment) is an exact whole-word
                 match in the message  (e.g. "blend" matches "FertilizerBlend")
          +10  — any camelCase segment matches a whole word in the message
           -1  — penalty per extra camelCase word (prefer simpler entity names)
        Whole-word matching avoids false substring hits (e.g. "end" inside "blend").
        """
        msg_lower = message.lower()
        best: tuple[float, str] | None = None

        for name in entities:
            parts = re.sub(
                r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name
            ).split()
            parts_lower = [p.lower() for p in parts]
            suffix = parts_lower[-1] if parts_lower else ""

            score: float = 0.0
            for w in parts_lower:
                if len(w) < 3:
                    continue
                if re.search(rf"\b{re.escape(w)}\b", msg_lower):
                    # Suffix match (e.g. "blend" in "FertilizerBlend") scores higher
                    score += 20 if w == suffix else 10

            # Penalise long names so "FertilizerBlend" (2 parts) beats
            # "LocationsForDefaultBatchSettings" (5 parts) on equal word score.
            if score > 0:
                score -= (len(parts) - 1) * 1.0

            if score > 0 and (best is None or score > best[0]):
                best = (score, name)

        return best[1] if best else None

    def _extract(self, message: str, app_id: str) -> dict | None:
        """Extract navigation parameters from *message*.

        Returns dict with keys: entity, key, view — or None when extraction fails.
        """
        entities = self._entity_names(app_id)
        entity = self._match_entity(message, entities) if entities else None

        key_match = _KEY_RE.search(message)
        key = key_match.group(1) if key_match else None

        is_list = bool(re.search(r"\blist\b|\ball\b", message, re.IGNORECASE)) and not key

        if not entity and not key:
            return None

        return {
            "entity": entity or "",
            "key": key or "",
            "view": "list" if is_list else "object",
        }

    # ── SSE helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    # ── Agent interface ───────────────────────────────────────────────────────

    async def stream_response(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        app_id: Optional[str] = None,
        fiori_context: Optional[Dict] = None,
        odata_token: Optional[str] = None,
        user_id: Optional[str] = None,
        raw_message: Optional[str] = None,
        backend_url: Optional[str] = None,
        prepared_context: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        t0 = time.monotonic()
        sse = self._sse

        if not app_id:
            yield sse({"type": "chunk", "content": "Navigation requires an app context."})
            yield sse({"type": "done", "model": "navigation-agent", "response_time": 0})
            return

        yield sse({"type": "exec_status", "step": "analyzing", "step_num": 1, "total_steps": 2})

        # Use the raw user text for extraction — `message` may have tool-call
        # context prepended by chat.py which would corrupt entity and key matching.
        nav = self._extract(raw_message or message, app_id)

        if not nav:
            yield sse({
                "type": "chunk",
                "content": (
                    "I couldn't determine which record to navigate to. "
                    "Please specify the entity and ID — for example: *open blend 1856*."
                ),
            })
            yield sse({"type": "done", "model": "navigation-agent",
                       "response_time": round(time.monotonic() - t0, 3)})
            return

        entity = nav["entity"]
        key    = nav["key"]
        view   = nav["view"]

        # Human-readable label for the chat bubble
        if entity and key:
            label = f"**{entity} {key}**"
        elif entity:
            label = f"**{entity}** list"
        else:
            label = f"record **{key}**"

        yield sse({
            "type": "exec_status", "step": "navigating",
            "step_num": 2, "total_steps": 2,
            "message": f"Opening {label.replace('**', '')}…",
        })
        yield sse({"type": "chunk", "content": f"\n\nOpening {label}."})
        yield sse({
            "type": "tool_result",
            "success": True,
            "tool_key": "btp.navigate",
            "execution_type": "UI_ACTION",
            "frontend_event": "BTP_NAVIGATE",
            "payload": {
                "entity": entity,
                "key":    key,
                "view":   view,
                "appId":  app_id,
            },
        })
        yield sse({
            "type": "done",
            "model": "navigation-agent",
            "response_time": round(time.monotonic() - t0, 3),
        })

    async def get_response(
        self,
        message: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Non-streaming fallback."""
        app_id = kwargs.get("app_id") or ""
        nav = self._extract(message, app_id) if app_id else None
        if nav:
            return {
                "response": f"Navigating to {nav['entity']} {nav['key']}.",
                "model": "navigation-agent",
                "navigation": nav,
            }
        return {
            "response": "I couldn't determine the navigation target.",
            "model": "navigation-agent",
        }

    def get_status(self) -> Dict[str, Any]:
        return {"agent_type": "NavigationAgent", "status": "healthy", "model": "rule-based"}
