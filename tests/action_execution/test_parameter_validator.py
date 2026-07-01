"""Unit tests — ParameterValidator type checking and required-field enforcement."""
from tests.action_execution.conftest import make_param, make_tool
from app.services.action_execution.parameter_validator import ParameterValidator

_v = ParameterValidator()


# ── required parameters ───────────────────────────────────────────────────────

def test_missing_required_parameter():
    tool = make_tool(
        parameters=[make_param("OrderID", "UUID", required=True)],
        required_parameters=["OrderID"],
    )
    result = _v.validate(tool, {})
    assert not result.valid
    assert any(e.field == "OrderID" for e in result.errors)


def test_all_required_present_passes():
    tool = make_tool(
        parameters=[make_param("OrderID", "UUID", required=True)],
        required_parameters=["OrderID"],
    )
    result = _v.validate(tool, {"OrderID": "550e8400-e29b-41d4-a716-446655440000"})
    assert result.valid
    assert result.errors == []


def test_optional_parameter_missing_is_valid():
    tool = make_tool(
        parameters=[make_param("Note", "String", required=False)],
        required_parameters=[],
    )
    result = _v.validate(tool, {})
    assert result.valid


# ── UUID ──────────────────────────────────────────────────────────────────────

def test_valid_uuid_accepted():
    tool = make_tool(parameters=[make_param("ID", "UUID")])
    result = _v.validate(tool, {"ID": "550e8400-e29b-41d4-a716-446655440000"})
    assert result.valid


def test_invalid_uuid_rejected():
    tool = make_tool(parameters=[make_param("ID", "UUID")])
    result = _v.validate(tool, {"ID": "not-a-uuid"})
    assert not result.valid
    assert result.errors[0].expected_type == "UUID"


# ── Integer ───────────────────────────────────────────────────────────────────

def test_valid_integer_accepted():
    tool = make_tool(parameters=[make_param("Qty", "Integer")])
    assert _v.validate(tool, {"Qty": 10}).valid
    assert _v.validate(tool, {"Qty": "42"}).valid   # string coercible to int


def test_boolean_rejected_as_integer():
    tool = make_tool(parameters=[make_param("Qty", "Integer")])
    result = _v.validate(tool, {"Qty": True})
    assert not result.valid
    assert "integer" in result.errors[0].message.lower()


def test_invalid_integer_rejected():
    tool = make_tool(parameters=[make_param("Qty", "Integer")])
    result = _v.validate(tool, {"Qty": "abc"})
    assert not result.valid


# ── Boolean ───────────────────────────────────────────────────────────────────

def test_bool_true_accepted():
    tool = make_tool(parameters=[make_param("IsActive", "Boolean")])
    assert _v.validate(tool, {"IsActive": True}).valid
    assert _v.validate(tool, {"IsActive": False}).valid
    assert _v.validate(tool, {"IsActive": "true"}).valid


def test_invalid_bool_rejected():
    tool = make_tool(parameters=[make_param("IsActive", "Boolean")])
    result = _v.validate(tool, {"IsActive": "yes"})
    assert not result.valid


# ── Decimal ───────────────────────────────────────────────────────────────────

def test_decimal_accepted():
    tool = make_tool(parameters=[make_param("Amount", "Decimal")])
    assert _v.validate(tool, {"Amount": 3.14}).valid
    assert _v.validate(tool, {"Amount": "3.14"}).valid
    assert _v.validate(tool, {"Amount": 0}).valid


def test_invalid_decimal_rejected():
    tool = make_tool(parameters=[make_param("Amount", "Decimal")])
    result = _v.validate(tool, {"Amount": "not-a-number"})
    assert not result.valid


# ── Date / DateTime ───────────────────────────────────────────────────────────

def test_valid_date_accepted():
    tool = make_tool(parameters=[make_param("PostingDate", "Date")])
    assert _v.validate(tool, {"PostingDate": "2024-06-30"}).valid


def test_invalid_date_rejected():
    tool = make_tool(parameters=[make_param("PostingDate", "Date")])
    result = _v.validate(tool, {"PostingDate": "30/06/2024"})
    assert not result.valid
    assert result.errors[0].expected_type == "Date"


def test_valid_datetime_accepted():
    tool = make_tool(parameters=[make_param("Ts", "DateTime")])
    assert _v.validate(tool, {"Ts": "2024-06-30T12:00:00Z"}).valid
    assert _v.validate(tool, {"Ts": "2024-06-30T12:00:00+02:00"}).valid


def test_invalid_datetime_rejected():
    tool = make_tool(parameters=[make_param("Ts", "DateTime")])
    result = _v.validate(tool, {"Ts": "June 30 2024"})
    assert not result.valid


# ── Collections ───────────────────────────────────────────────────────────────

def test_collection_accepted():
    tool = make_tool(parameters=[make_param("Tags", "String", is_collection=True)])
    assert _v.validate(tool, {"Tags": ["a", "b"]}).valid


def test_non_list_rejected_for_collection():
    tool = make_tool(parameters=[make_param("Tags", "String", is_collection=True)])
    result = _v.validate(tool, {"Tags": "not-a-list"})
    assert not result.valid
    assert result.errors[0].expected_type == "Array"


# ── Unknown / undeclared params ────────────────────────────────────────────────

def test_undeclared_param_passes_through():
    # Params not in the tool definition are not validated — CAP will reject them.
    tool = make_tool(parameters=[])
    result = _v.validate(tool, {"SomeExtraParam": "value"})
    assert result.valid


# ── None values ───────────────────────────────────────────────────────────────

def test_none_value_for_optional_param_is_ok():
    tool = make_tool(parameters=[make_param("Note", "String")], required_parameters=[])
    result = _v.validate(tool, {"Note": None})
    assert result.valid


def test_multiple_errors_all_reported():
    tool = make_tool(
        parameters=[
            make_param("ID", "UUID", required=True),
            make_param("Qty", "Integer", required=True),
        ],
        required_parameters=["ID", "Qty"],
    )
    result = _v.validate(tool, {})
    assert not result.valid
    assert len(result.errors) == 2
