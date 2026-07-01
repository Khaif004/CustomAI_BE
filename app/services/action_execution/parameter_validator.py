"""Validates parameters against a ToolDefinition's declared parameter schema.

All validation is local (no I/O, no LLM).  Returns structured errors so
callers can surface meaningful messages to the user.
"""
from __future__ import annotations

import re
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from app.models.tool_catalog import ToolDefinition
from app.services.action_execution.models import ValidationFieldError, ValidationResult

# ── Type-name normalisation sets (lower-case OData / CDS names) ───────────────

_UUID_TYPES = frozenset({"uuid", "guid", "cds.uuid"})
_INT_TYPES = frozenset({"integer", "int", "int32", "int64", "cds.integer", "cds.integer64"})
_BOOL_TYPES = frozenset({"boolean", "bool", "cds.boolean"})
_DECIMAL_TYPES = frozenset({"decimal", "number", "double", "float", "single",
                             "cds.decimal", "cds.double"})
_DATE_TYPES = frozenset({"date", "cds.date"})
_DATETIME_TYPES = frozenset({"datetime", "datetimeoffset", "timestamp",
                              "edm.datetimeoffset", "cds.datetime", "cds.timestamp"})

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
)


class ParameterValidator:
    """Validates parameters against ToolDefinition.parameters schema.

    Usage::
        result = ParameterValidator().validate(tool, {"OrderID": "some-uuid"})
        if not result.valid:
            raise ParameterValidationError(result.errors)
    """

    def validate(
        self,
        tool: ToolDefinition,
        parameters: Dict[str, Any],
    ) -> ValidationResult:
        errors: List[ValidationFieldError] = []

        # Build a fast lookup of declared parameter definitions
        declared = {p.name: p for p in (tool.parameters or [])}

        # 1. Required parameters must be present and non-None
        for req_name in (tool.required_parameters or []):
            if parameters.get(req_name) is None:
                p = declared.get(req_name)
                errors.append(ValidationFieldError(
                    field=req_name,
                    message=f"Required parameter '{req_name}' is missing.",
                    expected_type=p.type if p else None,
                ))

        # 2. Type-check every supplied non-None parameter that has a declaration
        for name, value in parameters.items():
            if value is None:
                continue
            p = declared.get(name)
            if p is None:
                continue  # undeclared params are passed through; CAP will reject if invalid
            errors.extend(
                self._check_type(name, value, p.type, p.cds_type, p.is_collection)
            )

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    # ── dispatch ──────────────────────────────────────────────────────────────

    def _check_type(
        self,
        field: str,
        value: Any,
        otype: Optional[str],
        cds_type: Optional[str],
        is_collection: bool,
    ) -> List[ValidationFieldError]:
        norm = (otype or "").lower().strip()
        cds = (cds_type or "").lower().strip()

        if is_collection:
            if not isinstance(value, list):
                return [ValidationFieldError(
                    field=field,
                    message=f"'{field}' must be a list (is_collection=true).",
                    expected_type="Array",
                    received_value=type(value).__name__,
                )]
            errors: List[ValidationFieldError] = []
            for i, elem in enumerate(value):
                errors.extend(self._check_type(f"{field}[{i}]", elem, otype, cds_type, False))
            return errors

        if norm in _UUID_TYPES or cds in _UUID_TYPES:
            return self._check_uuid(field, value)
        if norm in _INT_TYPES or cds in _INT_TYPES:
            return self._check_int(field, value)
        if norm in _BOOL_TYPES or cds in _BOOL_TYPES:
            return self._check_bool(field, value)
        if norm in _DECIMAL_TYPES or cds in _DECIMAL_TYPES:
            return self._check_decimal(field, value)
        if norm in _DATE_TYPES or cds in _DATE_TYPES:
            return self._check_date(field, value)
        if norm in _DATETIME_TYPES or cds in _DATETIME_TYPES:
            return self._check_datetime(field, value)
        # String and unrecognised types: anything str-coercible is accepted
        return []

    # ── type checkers ─────────────────────────────────────────────────────────

    def _check_uuid(self, field: str, value: Any) -> List[ValidationFieldError]:
        try:
            uuid.UUID(str(value))
            return []
        except (ValueError, AttributeError):
            return [ValidationFieldError(
                field=field,
                message=f"'{field}' must be a valid UUID (e.g. '550e8400-e29b-41d4-a716-446655440000').",
                expected_type="UUID",
                received_value=repr(value)[:80],
            )]

    def _check_int(self, field: str, value: Any) -> List[ValidationFieldError]:
        if isinstance(value, bool):
            return [ValidationFieldError(
                field=field,
                message=f"'{field}' must be an integer, not a boolean.",
                expected_type="Integer",
                received_value=repr(value),
            )]
        try:
            int(value)
            return []
        except (TypeError, ValueError):
            return [ValidationFieldError(
                field=field,
                message=f"'{field}' must be an integer.",
                expected_type="Integer",
                received_value=repr(value)[:80],
            )]

    def _check_bool(self, field: str, value: Any) -> List[ValidationFieldError]:
        if isinstance(value, bool):
            return []
        if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
            return []
        return [ValidationFieldError(
            field=field,
            message=f"'{field}' must be a boolean (true/false).",
            expected_type="Boolean",
            received_value=repr(value)[:80],
        )]

    def _check_decimal(self, field: str, value: Any) -> List[ValidationFieldError]:
        try:
            Decimal(str(value))
            return []
        except InvalidOperation:
            return [ValidationFieldError(
                field=field,
                message=f"'{field}' must be a numeric (decimal/float) value.",
                expected_type="Decimal",
                received_value=repr(value)[:80],
            )]

    def _check_date(self, field: str, value: Any) -> List[ValidationFieldError]:
        if isinstance(value, str) and _ISO_DATE_RE.match(value):
            return []
        return [ValidationFieldError(
            field=field,
            message=f"'{field}' must be an ISO 8601 date string (YYYY-MM-DD).",
            expected_type="Date",
            received_value=repr(value)[:80],
        )]

    def _check_datetime(self, field: str, value: Any) -> List[ValidationFieldError]:
        if isinstance(value, str) and _ISO_DATETIME_RE.match(value):
            return []
        return [ValidationFieldError(
            field=field,
            message=f"'{field}' must be an ISO 8601 datetime string.",
            expected_type="DateTime",
            received_value=repr(value)[:80],
        )]
