"""
/api/apps — App context registration endpoint.

Any CF app (e.g. Stutsman) calls POST /api/apps/register at startup
(or whenever its schema/data changes) to push its context into the
BTP Copilot vector store. Subsequent chat requests that include
`app_id` will retrieve this context and inject it into the LLM prompt.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from app.auth.security import get_current_user
from app.knowledge.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/apps", tags=["apps"])


class AppDocument(BaseModel):
    title: str = Field(..., description="Short label, e.g. 'SalesOrder entity schema'")
    content: str = Field(..., description="Plain-text content: schema, rules, relationships, etc.")


class AppRegistrationRequest(BaseModel):
    app_id: str = Field(
        ...,
        description="Stable, unique identifier for the app, e.g. 'stutsman'",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    app_name: str = Field(..., description="Human-readable name, e.g. 'Stutsman Sales App'")
    documents: List[AppDocument] = Field(
        ...,
        description="Context documents: entity schemas, relationship descriptions, business rules, etc.",
        min_length=1,
        max_length=50,
    )
    replace: bool = Field(True, description="Replace previously registered documents for this app_id")


class AppRegistrationResponse(BaseModel):
    app_id: str
    app_name: str
    chunks_stored: int
    docs_received: int
    message: str


@router.post("/register", response_model=AppRegistrationResponse, status_code=status.HTTP_200_OK)
async def register_app_context(
    request: AppRegistrationRequest,
    current_user=Depends(get_current_user),
):
    """
    Register or update the context for a host application.

    Call this from your app's startup or CI/CD pipeline whenever the
    schema or business rules change. The content is chunked, embedded,
    and stored in the vector store under the app's `app_id`.

    Example payload from Stutsman app:
    ```json
    {
      "app_id": "stutsman",
      "app_name": "Stutsman Sales App",
      "documents": [
        {
          "title": "SalesOrder entity",
          "content": "SalesOrder has fields: id, customerId, createdAt, status (OPEN/CLOSED), totalAmount..."
        },
        {
          "title": "ProcessOrder entity",
          "content": "ProcessOrder has fields: id, salesOrderId (FK), warehouseId, pickedAt, shippedAt..."
        },
        {
          "title": "SalesOrder to ProcessOrder relationship",
          "content": "Each SalesOrder can have one or more ProcessOrders. A SalesOrder transitions to CLOSED only when all its ProcessOrders reach status SHIPPED. ProcessOrder.salesOrderId is a foreign key referencing SalesOrder.id..."
        }
      ]
    }
    ```
    """
    try:
        kb = get_knowledge_base()
        result = kb.register_app_context(
            app_id=request.app_id,
            app_name=request.app_name,
            documents=[{"title": d.title, "content": d.content} for d in request.documents],
            replace=request.replace,
        )
        return AppRegistrationResponse(
            app_id=request.app_id,
            app_name=request.app_name,
            chunks_stored=result["chunks_stored"],
            docs_received=result["docs_received"],
            message=f"Context for '{request.app_name}' registered successfully.",
        )
    except Exception as e:
        logger.error(f"App registration failed for '{request.app_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}",
        )


@router.delete("/{app_id}", status_code=status.HTTP_200_OK)
async def deregister_app(app_id: str, current_user=Depends(get_current_user)):
    """Remove all stored context for an app."""
    try:
        kb = get_knowledge_base()
        kb._delete_by_app_id(app_id)
        return {"app_id": app_id, "message": "Context removed."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
