"""Navigation relay — allows the chatbot (any origin) to trigger SAP UI5 router
navigation in a Fiori app running in a different browser tab.

Flow
----
1. Chatbot sends  POST /api/navigation/pending   {app_id, entity, key, view, ...}
2. Widget polls   GET  /api/navigation/pending/{app_id}
   → Returns the pending event and removes it (one-shot delivery).
   → Returns {} when nothing is pending.

No authentication is required on these endpoints:
  - The POST is called by the chatbot which is already authenticated on the chat
    endpoint.  The navigation payload contains no sensitive data (entity names
    and record keys that the user explicitly typed).
  - The GET is called by the widget running in the host Fiori page — it cannot
    supply user credentials.

Events are stored in process memory (dict). This is intentional:
  - Navigation events are ephemeral (consume-once, TTL ~10 s).
  - A process restart drops any un-consumed event, which is harmless.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter(prefix="/api/navigation", tags=["navigation"])

# { app_id: { event dict + "_stored_at" timestamp } }
_pending: Dict[str, Dict[str, Any]] = {}

_TTL_SECONDS = 30  # drop stale events that were never consumed


def _purge_stale() -> None:
    now = time.monotonic()
    stale = [k for k, v in _pending.items() if now - v.get("_stored_at", 0) > _TTL_SECONDS]
    for k in stale:
        del _pending[k]


@router.post("/pending", status_code=204)
async def store_pending(event: Dict[str, Any]) -> None:
    """Store a navigation event for an app.  Overwrites any previous pending event."""
    app_id = event.get("app_id")
    if not app_id:
        return
    _purge_stale()
    _pending[str(app_id)] = {**event, "_stored_at": time.monotonic()}


@router.get("/pending/{app_id}")
async def consume_pending(app_id: str) -> Dict[str, Any]:
    """Return and remove the pending navigation event for *app_id*.
    Returns an empty dict {} when nothing is pending."""
    _purge_stale()
    event = _pending.pop(app_id, None)
    if not event:
        return {}
    event.pop("_stored_at", None)
    return event
