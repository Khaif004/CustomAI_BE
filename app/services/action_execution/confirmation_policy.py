"""Determines whether a tool execution requires explicit user confirmation.

Rule-based, no LLM, no network.  Checks the tool's name/key for mutating
verb fragments.  Used by the Widget to decide whether to show a confirmation
dialog before executing — but no UI lives here.
"""
from __future__ import annotations

from app.models.tool_catalog import ToolDefinition, ToolType

# Verb/keyword fragments (lower-case) that signal a destructive or
# irreversible state change.  Checked against display_name, name, tool_key,
# and cds_name as a combined string.
_CONFIRM_FRAGMENTS = frozenset({
    "release", "cancel", "delete", "activate", "deactivate",
    "post", "close", "reverse", "reject", "approve",
    "submit", "confirm", "complete", "finalize", "finalise",
    "lock", "unlock", "archive", "restore", "purge",
    "clear", "reset", "rollback", "undo", "force",
    "remove", "destroy", "discard", "terminate", "abort",
})

# These tool types never require confirmation: read-only types and UI_ACTION
# (which is a client-side event dispatch — no server-side state change).
_READ_ONLY_TYPES = frozenset({
    ToolType.FUNCTION, ToolType.NAVIGATION, ToolType.REPORT, ToolType.UI_ACTION,
})


class ConfirmationPolicy:
    """Return whether a registered tool requires explicit confirmation."""

    def requires_confirmation(self, tool: ToolDefinition) -> bool:
        """True iff the Widget should ask the user to confirm before executing.

        Logic:
        * FUNCTION / NAVIGATION / REPORT → always False (read-only).
        * For ACTIONs and others: check whether any _CONFIRM_FRAGMENTS appear
          anywhere in the combined identifier strings (case-insensitive).
        """
        if tool.tool_type in _READ_ONLY_TYPES:
            return False

        combined = " ".join(filter(None, [
            tool.tool_key or "",
            tool.name or "",
            tool.display_name or "",
            tool.cds_name or "",
        ])).lower()

        return any(frag in combined for frag in _CONFIRM_FRAGMENTS)
