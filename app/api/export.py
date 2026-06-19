import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from app.services.export_store import ExportStore
from app.services.export_service import generate_excel, generate_csv, generate_pdf, generate_word

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/export", tags=["export"])

FORMATS = {
    "excel": {
        "fn": generate_excel,
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "ext": "xlsx",
    },
    "csv": {
        "fn": generate_csv,
        "mime": "text/csv",
        "ext": "csv",
    },
    "pdf": {
        "fn": generate_pdf,
        "mime": "application/pdf",
        "ext": "pdf",
    },
    "word": {
        "fn": generate_word,
        "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "ext": "docx",
    },
}


class PrepareExportRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="Full list of records")
    entity: str = Field(..., description="Entity name, e.g. SalesOrder")
    fields: List[str] = Field(..., description="Ordered list of column names")
    total_count: Optional[int] = None


class PrepareExportResponse(BaseModel):
    key: str
    total_rows: int
    download_urls: Dict[str, str]


@router.post("/prepare", response_model=PrepareExportResponse)
async def prepare_export(request: PrepareExportRequest):
    """
    Store a dataset for later download.  Called by the agent when a live-data
    fetch returns more rows than the display limit.

    Returns a short-lived key + download URLs for each supported format.
    """
    key = ExportStore.put(
        rows=request.rows,
        entity=request.entity,
        fields=request.fields,
        total_count=request.total_count,
    )
    download_urls = {
        fmt: f"/api/export/{key}/{fmt}" for fmt in FORMATS
    }
    return PrepareExportResponse(
        key=key,
        total_rows=len(request.rows),
        download_urls=download_urls,
    )


@router.get("/{key}/{fmt}")
async def download_export(key: str, fmt: str):
    """
    Download the stored dataset in the requested format.
    Key expires after 30 minutes.
    """
    if fmt not in FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{fmt}'. Choose from: {', '.join(FORMATS)}",
        )

    entry = ExportStore.get(key)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export link has expired or is invalid. Please re-ask your question to generate a new one.",
        )

    fmt_cfg = FORMATS[fmt]
    entity = entry["entity"]
    rows = entry["rows"]
    fields = entry["fields"]

    total_count = entry.get("total_count", len(rows))

    try:
        if fmt == "csv":
            file_bytes = fmt_cfg["fn"](rows, fields)
        else:
            # excel, pdf, word all accept (rows, fields, entity_name, total_count)
            file_bytes = fmt_cfg["fn"](rows, fields, entity, total_count)
    except Exception as e:
        logger.error(f"Export generation failed ({fmt}, entity={entity}): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Export generation failed. Please try again.",
        )

    safe_entity = "".join(c if c.isalnum() else "_" for c in entity)
    filename = f"{safe_entity}.{fmt_cfg['ext']}"

    return Response(
        content=file_bytes,
        media_type=fmt_cfg["mime"],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
