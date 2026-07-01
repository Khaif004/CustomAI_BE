"""Unit tests — ConfirmationPolicy rule-based decisions."""
from app.models.tool_catalog import ToolType
from app.services.action_execution.confirmation_policy import ConfirmationPolicy
from tests.action_execution.conftest import make_tool

_p = ConfirmationPolicy()


# ── Mutating actions → True ───────────────────────────────────────────────────

def test_release_action_requires_confirmation():
    tool = make_tool(tool_key="ReleaseProcessOrder", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is True


def test_cancel_action_requires_confirmation():
    tool = make_tool(tool_key="CancelBlend", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is True


def test_delete_action_requires_confirmation():
    tool = make_tool(tool_key="DeleteSalesOrder", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is True


def test_activate_action_requires_confirmation():
    tool = make_tool(tool_key="ActivateWorkflow", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is True


def test_approve_requires_confirmation():
    tool = make_tool(tool_key="ApproveLeaveRequest", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is True


def test_reject_requires_confirmation():
    tool = make_tool(tool_key="RejectPurchaseOrder", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is True


def test_post_document_requires_confirmation():
    tool = make_tool(tool_key="PostGoodsReceipt", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is True


def test_close_order_requires_confirmation():
    tool = make_tool(tool_key="CloseServiceOrder", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is True


def test_reverse_document_requires_confirmation():
    tool = make_tool(tool_key="ReverseAcctDocument", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is True


# ── Read-only functions → False ───────────────────────────────────────────────

def test_function_never_requires_confirmation():
    tool = make_tool(tool_key="ReleaseProcessOrder", tool_type=ToolType.FUNCTION)
    assert _p.requires_confirmation(tool) is False


def test_navigation_never_requires_confirmation():
    tool = make_tool(tool_key="DeleteSalesOrder", tool_type=ToolType.NAVIGATION)
    assert _p.requires_confirmation(tool) is False


def test_report_never_requires_confirmation():
    tool = make_tool(tool_key="CancelReport", tool_type=ToolType.REPORT)
    assert _p.requires_confirmation(tool) is False


# ── Read-only actions (no mutating verb) → False ──────────────────────────────

def test_read_action_no_confirmation():
    tool = make_tool(tool_key="GetProcessOrderDetails", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is False


def test_list_action_no_confirmation():
    tool = make_tool(tool_key="ListOpenOrders", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is False


def test_create_action_no_confirmation():
    # "create" is not in the fragment list by design — creation is not destructive.
    tool = make_tool(tool_key="CreateDraftOrder", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is False


# ── Fragment matching is case-insensitive ─────────────────────────────────────

def test_uppercase_fragment_detected():
    tool = make_tool(tool_key="RELEASEORDER", tool_type=ToolType.ACTION)
    assert _p.requires_confirmation(tool) is True


def test_fragment_in_display_name():
    tool = make_tool(
        tool_key="com_sap_action_42",
        tool_type=ToolType.ACTION,
    )
    # Override display_name after construction
    tool.display_name = "Cancel Blend Order"
    assert _p.requires_confirmation(tool) is True
