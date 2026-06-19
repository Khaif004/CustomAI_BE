"""
export_service.py — Thin adapter that converts live OData row data into downloadable files.

Uses the same professional document builders as documents.py so all exports share
one rendering code path and one visual style.  No duplication.

Supported formats: excel, csv, pdf, word
"""

import csv
import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Shared conversion: OData rows → document-builder JSON structure ────────────

def _to_document_data(
    rows: List[Dict[str, Any]],
    fields: List[str],
    entity_name: str,
    total_count: int,
) -> dict:
    """Convert flat OData rows to the structured dict expected by documents.py builders."""
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    subtitle = (
        f"Exported {len(rows)} of {total_count} records  •  Generated {generated_at}"
        if total_count > len(rows) else
        f"{len(rows)} record{'s' if len(rows) != 1 else ''}  •  Generated {generated_at}"
    )
    table_rows = [[str(row.get(f, "")) for f in fields] for row in rows]
    return {
        "title": entity_name,
        "subtitle": subtitle,
        "sections": [
            {
                "heading": "Data Export",
                "level": 1,
                "content": (
                    f"This export contains {len(rows)} record(s) from the {entity_name} entity."
                    + (
                        f" {total_count - len(rows)} additional record(s) exist in the system."
                        if total_count > len(rows) else ""
                    )
                ),
                "bullets": None,
                "table": {"headers": fields, "rows": table_rows},
            }
        ],
        "conclusion": None,
        # Excel builder reads "sheets" key
        "sheets": [
            {
                "name": entity_name[:31],
                "description": f"Live OData data  •  {generated_at}",
                "headers": fields,
                "rows": table_rows,
                "summary": (
                    f"Showing {len(rows)} of {total_count} total records"
                    if total_count > len(rows) else None
                ),
            }
        ],
    }


# ── Format generators ──────────────────────────────────────────────────────────

def generate_csv(rows: List[Dict[str, Any]], fields: List[str]) -> bytes:
    """CSV with UTF-8 BOM (opens correctly in Excel on all platforms)."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=fields, extrasaction="ignore", lineterminator="\r\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({f: row.get(f, "") for f in fields})
    return b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")


def generate_excel(
    rows: List[Dict[str, Any]],
    fields: List[str],
    entity_name: str = "Data",
    total_count: Optional[int] = None,
) -> bytes:
    """Professional .xlsx — same builder and visual style as documents.py."""
    from app.api.documents import _build_excel
    data = _to_document_data(rows, fields, entity_name, total_count or len(rows))
    return _build_excel(data)


def generate_pdf(
    rows: List[Dict[str, Any]],
    fields: List[str],
    entity_name: str = "Data",
    total_count: Optional[int] = None,
) -> bytes:
    """Professional PDF — same builder and visual style as documents.py."""
    from app.api.documents import _build_pdf
    data = _to_document_data(rows, fields, entity_name, total_count or len(rows))
    return _build_pdf(data)


def generate_word(
    rows: List[Dict[str, Any]],
    fields: List[str],
    entity_name: str = "Data",
    total_count: Optional[int] = None,
) -> bytes:
    """Professional .docx — same builder and visual style as documents.py."""
    from app.api.documents import _build_word
    data = _to_document_data(rows, fields, entity_name, total_count or len(rows))
    return _build_word(data)

